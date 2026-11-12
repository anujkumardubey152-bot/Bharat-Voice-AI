import os
import re
import time

import requests
import streamlit as st
from google import genai
from google.genai import types


st.set_page_config(
    page_title="Bharat Voice AI",
    page_icon="🎙️",
    layout="centered",
)

# -------------------- Configuration --------------------

def get_secret(name):
    """Read a setting from Streamlit secrets, then environment variables."""
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        # Streamlit secrets may be unavailable when running outside deployment.
        pass
    return os.getenv(name)


GNANI_API_KEY = get_secret("GNANI_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")

# Keep the existing provider endpoints and model names.
GEMINI_MODEL = "gemini-3.8-flash"
STT_URL = "https://api.vachana.ai/stt/v3"
TTS_URL = "https://api.vachana.ai/api/v1/tts/inference"

# Individual network timeouts only. There is deliberately NO overall
# pipeline deadline: STT, Gemini, and TTS each get their own request window.
STT_TIMEOUT = (5, 35)          # connect timeout, read timeout (seconds)
GEMINI_TIMEOUT_MS = 35_000     # Google Gen AI SDK request timeout (milliseconds)
TTS_TIMEOUT = (5, 40)          # connect timeout, read timeout (seconds)

# Only retry transient Gemini failures. Authentication, permission, and
# invalid-request errors are not retried.
GEMINI_MAX_ATTEMPTS = 2
GEMINI_RETRY_BASE_SECONDS = 0.5
GEMINI_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key):
    """Reuse one Gemini client across Streamlit reruns for this API key."""
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
    )


def get_error_status_code(error):
    """Best-effort extraction across Google Gen AI SDK error versions."""
    for attribute in ("status_code", "code"):
        value = getattr(error, attribute, None)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass

    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def is_retryable_gemini_error(error):
    """Return True only for temporary overload/network/timeout failures."""
    status_code = get_error_status_code(error)
    if status_code in GEMINI_RETRYABLE_STATUS_CODES:
        return True

    # SDK transport exceptions can vary by installed SDK/httpx version.
    # Do not retry arbitrary API errors: retry only recognizable transport
    # and timeout exceptions when no HTTP status is available.
    if status_code is None:
        class_name = error.__class__.__name__.lower()
        module_name = error.__class__.__module__.lower()
        message = str(error).lower()

        transport_markers = (
            "timeout",
            "connecterror",
            "connectionerror",
            "networkerror",
        )
        if any(marker in class_name for marker in transport_markers):
            return True
        if "httpx" in module_name and any(
            marker in class_name for marker in ("timeout", "connect")
        ):
            return True
        if isinstance(error, (TimeoutError, ConnectionError)):
            return True
        if any(marker in message for marker in (
            "connection reset",
            "connection refused",
            "temporarily unavailable",
            "timed out",
        )):
            return True

    return False


# -------------------- Language routing --------------------

def choose_tts_language(text, detected):
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))

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


# -------------------- Gnani Prisma STT --------------------

def transcribe_audio(audio_bytes, filename):
    if not GNANI_API_KEY:
        raise RuntimeError(
            "GNANI_API_KEY is missing. Add it to Streamlit Secrets "
            "or the GNANI_API_KEY environment variable."
        )
    if not audio_bytes:
        raise ValueError("The selected audio file is empty. Please try another file.")

    try:
        response = requests.post(
            STT_URL,
            headers={"X-API-Key-ID": GNANI_API_KEY},
            files={
                "audio_file": (
                    filename,
                    audio_bytes,
                    "application/octet-stream",
                )
            },
            data={
                "model": "prisma-v3.0",
                "language_code": "indic-auto",
            },
            timeout=STT_TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout as error:
        raise RuntimeError(
            "Gnani Prisma speech recognition timed out. Please retry or use "
            "a shorter recording."
        ) from error
    except requests.exceptions.ConnectionError as error:
        raise RuntimeError(
            "Could not connect to the Gnani speech recognition service. "
            "Check your internet connection and try again."
        ) from error
    except requests.exceptions.HTTPError as error:
        detail = ""
        if error.response is not None:
            detail = error.response.text[:1000]
        raise RuntimeError(
            f"Gnani STT request failed ({error}). {detail}"
        ) from error
    except requests.exceptions.RequestException as error:
        raise RuntimeError(f"Gnani STT network request failed: {error}") from error

    try:
        result = response.json()
    except ValueError as error:
        raise RuntimeError("Gnani STT returned an invalid JSON response.") from error

    if not isinstance(result, dict):
        raise RuntimeError("Gnani STT returned an unexpected response format.")

    transcript = result.get("transcript", "")
    detected = result.get("detected_language_code", "unknown")
    if not isinstance(transcript, str):
        raise RuntimeError("Gnani STT returned an invalid transcript format.")

    return transcript, detected


# -------------------- Gemini response --------------------

def generate_answer(user_text):
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to Streamlit Secrets "
            "or the GEMINI_API_KEY environment variable."
        )

    client = get_gemini_client(GEMINI_API_KEY)

    # Keep voice answers concise to reduce generation latency and TTS time.
    prompt = (
        "You are Bharat Voice AI, a helpful multilingual Indian voice assistant. "
        "Answer accurately in the user's language and style (Hindi, English, "
        "or natural Hinglish). Keep spoken replies to 1-3 concise sentences. "
        "Do not invent facts; state uncertainty when needed.\n\n"
        f"User: {user_text[:2500]}"
    )

    for attempt in range(1, GEMINI_MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=120,
                ),
            )
            answer = (response.text or "").strip()
            if not answer:
                raise ValueError(
                    "Gemini returned an empty response. Please try again."
                )
            return answer

        except ValueError:
            # Empty model output is not a transient API failure.
            raise
        except Exception as error:
            if (
                attempt < GEMINI_MAX_ATTEMPTS
                and is_retryable_gemini_error(error)
            ):
                # Small exponential backoff only for a classified transient
                # failure; there is no delay on successful requests.
                time.sleep(GEMINI_RETRY_BASE_SECONDS * attempt)
                continue

            status_code = get_error_status_code(error)
            if status_code in (401, 403):
                raise RuntimeError(
                    "Gemini authentication/permission failed. Check GEMINI_API_KEY "
                    "and whether it has access to the configured model."
                ) from error
            if status_code == 429:
                raise RuntimeError(
                    "Gemini is temporarily rate-limited or overloaded. "
                    "Please try again shortly."
                ) from error
            if status_code in (500, 502, 503, 504):
                raise RuntimeError(
                    "Gemini is temporarily unavailable. Please try again shortly."
                ) from error
            if is_retryable_gemini_error(error):
                raise RuntimeError(
                    "A temporary network or timeout error occurred while contacting "
                    "Gemini. Please try again."
                ) from error
            raise RuntimeError(f"Gemini response generation failed: {error}") from error

    raise RuntimeError("Gemini could not generate a response after retrying.")


# -------------------- Gnani Timbre TTS --------------------

def generate_speech(text, language):
    if not GNANI_API_KEY:
        raise RuntimeError(
            "GNANI_API_KEY is missing. Add it to Streamlit Secrets "
            "or the GNANI_API_KEY environment variable."
        )
    if not text.strip():
        raise ValueError("There is no AI response text to convert into speech.")

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
            timeout=TTS_TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout as error:
        raise RuntimeError(
            "Gnani Timbre voice generation timed out. Your text response is "
            "available above; please retry voice generation."
        ) from error
    except requests.exceptions.ConnectionError as error:
        raise RuntimeError(
            "Could not connect to Gnani Timbre. Check your internet connection "
            "and try again."
        ) from error
    except requests.exceptions.HTTPError as error:
        detail = ""
        if error.response is not None:
            detail = error.response.text[:1000]
        raise RuntimeError(
            f"Gnani TTS request failed ({error}). {detail}"
        ) from error
    except requests.exceptions.RequestException as error:
        raise RuntimeError(f"Gnani TTS network request failed: {error}") from error

    if not response.content:
        raise RuntimeError("Gnani Timbre returned an empty audio response.")
    return response.content


# -------------------- Streamlit interface --------------------

st.title("🎙️ Bharat Voice AI")
st.caption("Every Indian Voice. One Intelligent Platform.")

st.markdown(
    "Speak naturally. Bharat Voice AI listens, understands, "
    "and responds with a generated voice."
)

if not GNANI_API_KEY or not GEMINI_API_KEY:
    missing_keys = []
    if not GNANI_API_KEY:
        missing_keys.append("GNANI_API_KEY")
    if not GEMINI_API_KEY:
        missing_keys.append("GEMINI_API_KEY")
    st.error(
        "Missing API key(s): "
        + ", ".join(missing_keys)
        + ". Add them to Streamlit Secrets or environment variables."
    )
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

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

if st.button("Process my voice", type="primary", use_container_width=True):
    if audio_file is None:
        st.warning("Please record or upload audio first.")
    else:
        pipeline_start = time.perf_counter()
        timings = {}

        try:
            # Stage 1: STT
            with st.status("1/3 — Recognizing your speech...", expanded=True) as status:
                stage_start = time.perf_counter()
                audio_bytes = audio_file.getvalue()
                if not audio_bytes:
                    raise ValueError(
                        "The selected audio is empty. Please record or upload it again."
                    )
                transcript, detected = transcribe_audio(
                    audio_bytes,
                    getattr(audio_file, "name", None) or "recording.wav",
                )
                timings["STT"] = time.perf_counter() - stage_start
                status.update(label="1/3 — Speech recognized", state="complete")

            if not transcript.strip():
                st.warning("No speech was detected. Please try recording again.")
            else:
                st.subheader("You said")
                st.write(transcript)
                st.caption(f"Detected language: {detected}")

                # Stage 2: Gemini
                with st.status(
                    "2/3 — Generating the AI response...", expanded=True
                ) as status:
                    stage_start = time.perf_counter()
                    answer = generate_answer(transcript)
                    timings["Gemini"] = time.perf_counter() - stage_start
                    status.update(
                        label="2/3 — AI response ready", state="complete"
                    )

                st.subheader("Bharat Voice AI says")
                st.write(answer)

                # Stage 3: TTS
                tts_language = choose_tts_language(transcript, detected)
                with st.status(
                    "3/3 — Generating your voice response...", expanded=True
                ) as status:
                    stage_start = time.perf_counter()
                    audio_output = generate_speech(answer, tts_language)
                    timings["TTS"] = time.perf_counter() - stage_start
                    status.update(
                        label="3/3 — Voice response ready", state="complete"
                    )

                st.subheader("Listen to the response")
                st.audio(audio_output, format="audio/wav")

                st.session_state.history.append({
                    "user": transcript,
                    "assistant": answer,
                })

                total_time = time.perf_counter() - pipeline_start
                st.success("Conversation completed successfully!")

                st.subheader("Performance")
                cols = st.columns(4)
                cols[0].metric("Speech recognition", f"{timings['STT']:.1f}s")
                cols[1].metric("AI response", f"{timings['Gemini']:.1f}s")
                cols[2].metric("Voice generation", f"{timings['TTS']:.1f}s")
                cols[3].metric("Total", f"{total_time:.1f}s")

        except ValueError as error:
            st.warning(str(error))
        except RuntimeError as error:
            st.error(str(error))
        except requests.exceptions.RequestException as error:
            st.error(f"Network request failed: {error}")
        except Exception as error:
            st.error(f"Unexpected processing error: {error}")


# -------------------- Conversation history --------------------

if st.session_state.history:
    st.divider()
    st.subheader("Conversation history")

    for index, item in enumerate(reversed(st.session_state.history), 1):
        with st.expander(f"Conversation {index}"):
            st.markdown("**You:**")
            st.write(item["user"])
            st.markdown("**Bharat Voice AI:**")
            st.write(item["assistant"])

    if st.button("Clear conversation history"):
        st.session_state.history = []
        st.rerun()

st.divider()
st.caption("Bharat Voice AI | Powered by Gnani Prisma, Gemini and Timbre")
                
