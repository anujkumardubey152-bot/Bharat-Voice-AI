import logging
import os
import re
import time
import wave
from io import BytesIO

import requests
import streamlit as st
from google import genai
from google.genai import types


# ============================================================
# Bharat Voice AI — deadline-aware, multilingual voice pipeline
# ============================================================

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Bharat Voice AI",
    page_icon="🎙️",
    layout="centered",
)

# -------------------- Configuration -------------------------

STT_URL = "https://api.vachana.ai/stt/v3"
TTS_URL = "https://api.vachana.ai/api/v1/tts/inference"

PRIMARY_GEMINI_MODEL = "gemini-3.8-flash"

TOTAL_PIPELINE_TIMEOUT_SECONDS = 120.0
STT_BUDGET_SECONDS = 45.0
GEMINI_BUDGET_SECONDS = 30.0
TTS_BUDGET_SECONDS = 40.0
MIN_STAGE_START_SECONDS = 1.0

RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
MAX_GEMINI_MODELS_PER_REQUEST = 4
MAX_AUDIO_BYTES = 25 * 1024 * 1024


def get_secret(name, default=None):
    """Read a secret from Streamlit Secrets, then environment variables."""
    try:
        value = st.secrets.get(name)
        if value not in (None, ""):
            return value
    except Exception:
        # Streamlit Secrets may not be configured in local development.
        pass
    return os.getenv(name, default)


GNANI_API_KEY = get_secret("GNANI_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")

# Comma-separated names supplied by the owner. These are NOT trusted
# until they are found in the authenticated account's models.list result.
GEMINI_FALLBACK_MODELS_RAW = get_secret("GEMINI_FALLBACK_MODELS", "")
GEMINI_FALLBACK_MODELS = [
    item.strip()
    for item in str(GEMINI_FALLBACK_MODELS_RAW).split(",")
    if item.strip()
]

# -------------------- Errors and timing --------------------


class PipelineTimeoutError(TimeoutError):
    """The shared end-to-end deadline or a stage budget was exhausted."""


class StageError(RuntimeError):
    """A user-facing failure associated with one pipeline stage."""

    def __init__(self, stage, message):
        super().__init__(message)
        self.stage = stage
        self.message = message


def remaining_time(deadline, stage_budget, stage_name):
    """Return the stage timeout, never exceeding the shared deadline."""
    remaining = deadline - time.monotonic()
    if remaining < MIN_STAGE_START_SECONDS:
        raise PipelineTimeoutError(
            f"Not enough time remains to safely start {stage_name}."
        )
    return min(float(stage_budget), remaining)


def check_deadline(deadline, stage_name):
    if deadline - time.monotonic() < MIN_STAGE_START_SECONDS:
        raise PipelineTimeoutError(
            f"The 120-second processing deadline was reached before {stage_name}."
        )


def elapsed_seconds(start):
    return max(0.0, time.monotonic() - start)


# -------------------- Cached Gemini client -----------------


@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key):
    """
    Reuse the SDK client safely.

    SDK retries are disabled: application-level fallback controls attempts.
    A per-call timeout is also supplied to generate_content.
    """
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=10_000,
            retry_options=types.HttpRetryOptions(
                attempts=1,
                initial_delay=0,
                max_delay=0,
                http_status_codes=[],
            ),
        ),
    )


def _model_name(model):
    name = getattr(model, "name", None)
    if not name:
        return ""
    return str(name).removeprefix("models/")


def discover_account_flash_models(client, deadline):
    """
    Discover Flash models exposed to this API key.

    A model is eligible only if:
      1. It appears in this account's models.list response.
      2. Its name contains 'flash'.
      3. Its supported_actions include generateContent, when that
         metadata is supplied by the API.

    This is discovery, not a guarantee of quota or availability.
    Successful generation is the final runtime verification.
    """
    timeout = remaining_time(deadline, GEMINI_BUDGET_SECONDS, "Gemini")
    discovered = []

    try:
        # The SDK's list operation is synchronous. The client timeout is
        # deliberately short so discovery cannot consume the whole stage.
        for model in client.models.list(config={"page_size": 100}):
            check_deadline(deadline, "Gemini model discovery")
            name = _model_name(model)
            if not name or "flash" not in name.lower():
                continue

            actions = getattr(model, "supported_actions", None)
            # If the API does not expose capability metadata, do not
            # advertise this model as a verified generateContent candidate.
            if not actions:
                continue

            normalized_actions = {
                str(action).replace("_", "").lower()
                for action in actions
            }
            if "generatecontent" not in normalized_actions:
                continue

            discovered.append(name)

            if len(discovered) >= 100:
                break

        return discovered, None

    except Exception as exc:
        # Discovery is helpful but must not prevent the primary model
        # from being attempted. Fallbacks are not used without discovery.
        return [], f"{type(exc).__name__}: {exc}"[:300]


def get_retryable_status(error):
    """Extract an HTTP status from Google SDK/API exceptions if available."""
    for attribute in ("code", "status_code", "http_status"):
        value = getattr(error, attribute, None)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass

    response = getattr(error, "response", None)
    for attribute in ("status_code", "status"):
        value = getattr(response, attribute, None)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass

    # Some SDK/network exceptions expose only a textual status.
    match = re.search(r"\b(429|500|502|503|504)\b", str(error))
    return int(match.group(1)) if match else None


def is_timeout_error(error):
    if isinstance(error, (TimeoutError, requests.exceptions.Timeout)):
        return True

    name = type(error).__name__.lower()
    message = str(error).lower()
    return (
        "timeout" in name
        or "timed out" in message
        or "timeout" in message
    )


def is_auth_or_request_error(error):
    status = get_retryable_status(error)
    if status in (400, 401, 403, 404, 405, 422):
        return True

    message = str(error).lower()
    non_retryable_phrases = (
        "api key not valid",
        "invalid api key",
        "unauthenticated",
        "permission denied",
        "permission_denied",
        "access denied",
        "invalid argument",
        "invalid model",
        "not found",
    )
    return any(phrase in message for phrase in non_retryable_phrases)


def generate_answer(user_text, deadline):
    """Generate a concise answer with bounded, account-discovered fallback."""
    # One shared 30-second Gemini-stage deadline covers discovery AND
    # every model attempt; fallback calls do not each receive a fresh 30s.
    gemini_deadline = min(
        deadline, time.monotonic() + GEMINI_BUDGET_SECONDS
    )
    remaining_time(
        gemini_deadline, GEMINI_BUDGET_SECONDS,
        "Gemini response generation"
    )

    client = get_gemini_client(GEMINI_API_KEY)

    prompt = (
        "You are Bharat Voice AI, a helpful multilingual Indian voice assistant. "
        "Answer accurately and naturally in the user's language and style. "
        "Preserve Hindi, English, and Hinglish. Keep the spoken response to "
        "2–4 concise sentences. Do not invent facts; clearly say when unsure.\n\n"
        f"User: {user_text[:4000]}"
    )

    # Discovery is attempted only when there is enough Gemini time.
    discovered, discovery_error = discover_account_flash_models(
        client, gemini_deadline
    )
    discovered_set = set(discovered)

    # Preserve the current primary even if model listing is temporarily
    # unavailable. Only discovered models can be used as fallbacks.
    candidates = [PRIMARY_GEMINI_MODEL]
    for configured_name in GEMINI_FALLBACK_MODELS:
        normalized = configured_name.removeprefix("models/")
        if (
            normalized in discovered_set
            and normalized != PRIMARY_GEMINI_MODEL
            and normalized not in candidates
        ):
            candidates.append(normalized)

    # If no configured fallback was supplied, use discovered Flash models
    # as candidates. They are still tested by a real generation request.
    if not GEMINI_FALLBACK_MODELS:
        for name in discovered:
            if name != PRIMARY_GEMINI_MODEL and name not in candidates:
                candidates.append(name)

    candidates = candidates[:MAX_GEMINI_MODELS_PER_REQUEST]
    failures = []

    for index, model_name in enumerate(candidates):
        # Do not start a model attempt unless time remains.
        remaining = remaining_time(
            gemini_deadline, GEMINI_BUDGET_SECONDS, "Gemini model attempt"
        )
        if remaining < MIN_STAGE_START_SECONDS:
            break

        # Reserve a small part of the Gemini stage for returning the result.
        per_attempt_timeout_ms = max(1000, int(remaining * 1000))

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.35,
                    max_output_tokens=180,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                    http_options=types.HttpOptions(
                        timeout=per_attempt_timeout_ms,
                        retry_options=types.HttpRetryOptions(
                            attempts=1,
                            initial_delay=0,
                            max_delay=0,
                            http_status_codes=[],
                        ),
                    ),
                ),
            )

            answer = (response.text or "").strip()
            if not answer:
                raise ValueError("Gemini returned an empty response.")

            logging.info("Gemini generation succeeded with model=%s", model_name)
            return (
                answer, model_name, failures, discovery_error,
                candidates, discovered,
            )

        except Exception as exc:
            status = get_retryable_status(exc)
            retryable = (
                is_timeout_error(exc)
                or status in RETRYABLE_HTTP_CODES
            )

            logging.warning(
                "Gemini model attempt failed: model=%s status=%s error=%s",
                model_name, status, str(exc)[:250],
            )
            failures.append({
                "model": model_name,
                "status": status,
                "error": f"{type(exc).__name__}: {exc}"[:350],
            })

            # Authentication, permission, invalid-request and unknown
            # non-transient errors must not trigger model hopping.
            if is_auth_or_request_error(exc) or not retryable:
                raise StageError(
                    "Gemini",
                    f"Gemini failed on {model_name}: {type(exc).__name__}: {exc}",
                ) from exc

            # Temporary overload/network failure: try the next eligible model.
            if index == len(candidates) - 1:
                break

    details = "; ".join(
        f"{item['model']} ({item['status'] or 'timeout/network'}): "
        f"{item['error']}"
        for item in failures
    )
    if not details:
        details = "No eligible fallback model was available."

    raise StageError(
        "Gemini",
        "All eligible Gemini attempts failed. "
        f"Primary model: {PRIMARY_GEMINI_MODEL}. Details: {details}",
    )


# -------------------- Language routing ---------------------


def choose_tts_language(text, detected):
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))

    # Preserve the existing mixed-language routing behavior.
    if has_devanagari and has_latin:
        return "auto"

    if detected == "hi-IN" and has_latin:
        return "auto"

    supported = {
        "hi-IN", "en-IN", "bn-IN", "gu-IN", "kn-IN",
        "ml-IN", "mr-IN", "pa-IN", "ta-IN", "te-IN",
    }

    if detected in supported:
        return detected
    if has_devanagari:
        return "hi-IN"
    return "en-IN"


# -------------------- Prisma speech recognition ------------


def transcribe_audio(audio_bytes, filename, deadline):
    timeout = remaining_time(deadline, STT_BUDGET_SECONDS, "Prisma STT")

    if not audio_bytes:
        raise StageError("STT", "The selected audio file is empty.")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise StageError(
            "STT",
            "Audio file exceeds the 25 MB application limit. "
            "Please upload a shorter or smaller recording.",
        )

    try:
        response = requests.post(
            STT_URL,
            headers={"X-API-Key-ID": GNANI_API_KEY},
            files={
                "audio_file": (
                    filename or "recording.wav",
                    audio_bytes,
                    "application/octet-stream",
                )
            },
            data={
                "model": "prisma-v3.0",
                "language_code": "indic-auto",
            },
            timeout=timeout,
        )
        response.raise_for_status()

        try:
            result = response.json()
        except ValueError as exc:
            raise StageError(
                "STT", "Prisma returned an invalid JSON response."
            ) from exc

        if not isinstance(result, dict):
            raise StageError("STT", "Prisma returned an unexpected response.")

        transcript = result.get("transcript")
        detected = result.get("detected_language_code", "unknown")

        if not isinstance(transcript, str):
            raise StageError(
                "STT", "Prisma response did not contain a valid transcript."
            )

        return transcript.strip(), str(detected or "unknown")

    except StageError:
        raise
    except requests.exceptions.Timeout as exc:
        raise StageError("STT", "Prisma speech recognition timed out.") from exc
    except requests.exceptions.RequestException as exc:
        raise StageError("STT", f"Prisma network/API error: {exc}") from exc


# -------------------- Timbre speech generation -------------


def validate_wav(audio_bytes):
    """Reject empty, malformed, or non-WAV audio before playback."""
    if not audio_bytes:
        raise ValueError("Timbre returned an empty audio response.")

    if len(audio_bytes) < 44 or audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
        raise ValueError(
            "Timbre response is not a valid RIFF/WAVE file. "
            "Check the provider's response format."
        )

    try:
        with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
            if wav_file.getnframes() <= 0:
                raise ValueError("WAV contains no audio frames.")
            if wav_file.getnchannels() <= 0 or wav_file.getframerate() <= 0:
                raise ValueError("WAV audio metadata is invalid.")
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"Invalid WAV audio: {exc}") from exc

    return audio_bytes


def generate_speech(text, language, deadline):
    timeout = remaining_time(deadline, TTS_BUDGET_SECONDS, "Timbre TTS")

    # Existing Timbre model, voice, endpoint, and payload format are preserved.
    payload = {
        "text": text,
        "model": "timbre-v3.0",
        "voice": "Shivani",
        "language": language,
        "speed": 1.0,
        "audio_config": {
            "encoding": "linear_pcm",
            "container": "wav",
            "num_channels": 1,
            "sample_rate": 48000,
            "sample_width": 2,
        },
    }

    try:
        response = requests.post(
            TTS_URL,
            headers={
                "X-API-Key-ID": GNANI_API_KEY,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return validate_wav(response.content)

    except StageError:
        raise
    except requests.exceptions.Timeout as exc:
        raise StageError("TTS", "Timbre voice generation timed out.") from exc
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        body = exc.response.text[:500] if exc.response is not None else ""
        raise StageError(
            "TTS", f"Timbre returned HTTP {status}. {body}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise StageError("TTS", f"Timbre network/API error: {exc}") from exc
    except ValueError as exc:
        raise StageError("TTS", str(exc)) from exc


# -------------------- Session state and interface -----------


if "history" not in st.session_state:
    st.session_state.history = []

if "last_processed_audio_id" not in st.session_state:
    st.session_state.last_processed_audio_id = None

if "processing_result" not in st.session_state:
    st.session_state.processing_result = None

st.title("🎙️ Bharat Voice AI")
st.caption("Every Indian Voice. One Intelligent Platform.")
st.markdown(
    "Speak naturally. Bharat Voice AI listens, understands "
    "and responds with a generated voice."
)

if not GNANI_API_KEY or not GEMINI_API_KEY:
    st.error(
        "API keys are missing. Configure GNANI_API_KEY and "
        "GEMINI_API_KEY in Streamlit Secrets or environment variables."
    )
    st.stop()

st.subheader("Talk to Bharat Voice AI")

audio_source = st.radio(
    "Choose your audio input",
    ["Record voice", "Upload audio"],
    horizontal=True,
)

if audio_source == "Record voice":
    audio_file = st.audio_input("Tap to record your voice")
else:
    audio_file = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "flac", "aac", "ogg", "m4a"],
    )

if audio_file is not None:
    st.caption(
        f"Selected audio: {audio_file.name} "
        f"({len(audio_file.getvalue()) / 1024:.1f} KB)"
    )

process_clicked = st.button(
    "Process my voice",
    type="primary",
    use_container_width=True,
    disabled=audio_file is None,
)

if process_clicked and audio_file is not None:
    audio_bytes = audio_file.getvalue()
    # A stable per-upload identity prevents accidental repeat processing
    # when Streamlit reruns without the selected audio changing.
    import hashlib
    audio_id = hashlib.sha256(audio_bytes).hexdigest()

    if audio_id == st.session_state.last_processed_audio_id:
        st.info(
            "This recording has already been processed. "
            "Record or upload new audio to start another request."
        )
    else:
        pipeline_start = time.monotonic()
        deadline = pipeline_start + TOTAL_PIPELINE_TIMEOUT_SECONDS
        timings = {}
        transcript = None
        answer = None
        audio_output = None
        successful_model = None
        detected_language = "unknown"
        current_stage = "Audio input"

        try:
            # Stage 1 — STT
            current_stage = "Speech recognition (Prisma STT)"
            with st.status(
                "1/3 — Listening and transcribing...",
                expanded=True,
            ) as status:
                stage_start = time.monotonic()
                st.write("Sending audio to Gnani Prisma v3.0.")
                transcript, detected_language = transcribe_audio(
                    audio_bytes,
                    audio_file.name or "recording.wav",
                    deadline,
                )
                timings["STT"] = elapsed_seconds(stage_start)

                if not transcript:
                    raise StageError(
                        "STT",
                        "No speech was detected. Please try recording again.",
                    )

                status.update(
                    label=f"1/3 — Transcription complete ({timings['STT']:.1f}s)",
                    state="complete",
                )

            st.subheader("You said")
            st.write(transcript)
            st.caption(f"Detected language: {detected_language}")

            # Stage 2 — Gemini
            check_deadline(deadline, "Gemini response generation")
            current_stage = "AI response (Gemini)"
            with st.status(
                "2/3 — Generating the AI response...",
                expanded=True,
            ) as status:
                stage_start = time.monotonic()
                st.write("Generating a concise multilingual response.")
                (
                    answer, successful_model, model_failures,
                    discovery_error, eligible_models, discovered_models,
                ) = generate_answer(transcript, deadline)
                timings["Gemini"] = elapsed_seconds(stage_start)

                status.write(f"Successful model: `{successful_model}`")
                status.write(
                    "Eligible model candidates: "
                    + (", ".join(eligible_models) if eligible_models else "primary only")
                )
                if discovered_models:
                    status.write(
                        f"Flash models discovered for this API key: "
                        f"{', '.join(discovered_models[:8])}"
                    )
                if discovery_error:
                    status.write(
                        "Model discovery was unavailable; only the primary "
                        "model could be safely attempted."
                    )
                status.update(
                    label=f"2/3 — AI response ready ({timings['Gemini']:.1f}s)",
                    state="complete",
                )

            st.subheader("Bharat Voice AI says")
            st.write(answer)
            st.caption(f"Gemini model used: {successful_model}")

            # Stage 3 — TTS
            check_deadline(deadline, "Timbre voice generation")
            current_stage = "Voice generation (Gnani Timbre)"
            tts_language = choose_tts_language(transcript, detected_language)

            with st.status(
                "3/3 — Creating the voice response...",
                expanded=True,
            ) as status:
                stage_start = time.monotonic()
                st.write(f"Using Gnani Timbre v3.0 — language: {tts_language}.")
                audio_output = generate_speech(answer, tts_language, deadline)
                timings["TTS"] = elapsed_seconds(stage_start)
                status.update(
                    label=f"3/3 — Voice response ready ({timings['TTS']:.1f}s)",
                    state="complete",
                )

            st.subheader("Listen to the response")
            st.audio(audio_output, format="audio/wav")

            st.session_state.history.append({
                "user": transcript,
                "assistant": answer,
                "language": detected_language,
                "model": successful_model,
            })
            st.session_state.last_processed_audio_id = audio_id
            st.session_state.processing_result = {
                "transcript": transcript,
                "answer": answer,
                "model": successful_model,
            }

            total_time = elapsed_seconds(pipeline_start)
            st.success("Conversation completed successfully!")

            st.subheader("Performance")
            cols = st.columns(4)
            cols[0].metric("STT", f"{timings['STT']:.1f}s")
            cols[1].metric("Gemini", f"{timings['Gemini']:.1f}s")
            cols[2].metric("TTS", f"{timings['TTS']:.1f}s")
            cols[3].metric("Total", f"{total_time:.1f}s")

            if total_time > TOTAL_PIPELINE_TIMEOUT_SECONDS:
                st.warning(
                    "The measured wall-clock duration exceeded 120 seconds. "
                    "Review provider/network latency; synchronous HTTP timeouts "
                    "are bounded but are not a hard real-time guarantee."
                )
            else:
                st.caption(
                    f"End-to-end processing: {total_time:.1f}s "
                    f"(120-second target)"
                )

        except StageError as exc:
            st.error(f"{exc.stage} failed: {exc.message}")
            st.info(
                "You can retry this recording. Check the relevant API status, "
                "network connection and configured credentials."
            )
            if exc.stage == "Gemini":
                st.caption(
                    "Gemini fallback attempts are limited to discovered Flash "
                    "models. Authentication and invalid-request errors are "
                    "not retried."
                )

        except PipelineTimeoutError as exc:
            st.error(f"Processing deadline reached: {exc}")
            st.info(
                "The next stage was not started because insufficient time "
                "remained. Try a shorter recording or retry later."
            )

        except Exception as exc:
            st.error(
                f"{current_stage} failed unexpectedly: "
                f"{type(exc).__name__}: {exc}"
            )
            st.info(
                "Please retry. If the problem persists, inspect the Streamlit "
                "Cloud application logs (do not share API keys)."
            )


# -------------------- Conversation history -----------------


if st.session_state.history:
    st.divider()
    st.subheader("Conversation history")

    for index, item in enumerate(reversed(st.session_state.history), 1):
        with st.expander(f"Conversation {index}"):
            st.markdown("**You:**")
            st.write(item["user"])
            st.markdown("**Bharat Voice AI:**")
            st.write(item["assistant"])
            st.caption(
                f"Language: {item.get('language', 'unknown')} · "
                f"Gemini model: {item.get('model', 'unknown')}"
            )

    if st.button("Clear conversation history"):
        st.session_state.history = []
        st.session_state.last_processed_audio_id = None
        st.session_state.processing_result = None
        st.rerun()

st.divider()
st.caption(
    "Bharat Voice AI | Powered by Gnani Prisma, Gemini and Timbre "
    "| 120-second processing target"
)
