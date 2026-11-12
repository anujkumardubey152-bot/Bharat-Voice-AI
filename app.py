import os
import re
import time

import streamlit as st
import requests
from google import genai
from google.genai import types


st.set_page_config(
    page_title="Bharat Voice AI",
    page_icon="🎙️",
    layout="centered"
)

# ---------- Configuration ----------

def get_secret(name):
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name)


GNANI_API_KEY = get_secret("GNANI_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")

GEMINI_MODEL = "gemini-3.8-flash"
STT_URL = "https://api.vachana.ai/stt/v3"
TTS_URL = "https://api.vachana.ai/api/v1/tts/inference"

# Timeouts are in milliseconds for the Gemini SDK and seconds for requests.
# Per-stage budgets add up to 115 seconds, leaving 5 seconds of overhead
# inside the 120-second end-to-end processing budget.
TOTAL_PIPELINE_TIMEOUT_SECONDS = 120
GNANI_STT_TIMEOUT_SECONDS = 45
GEMINI_TIMEOUT_SECONDS = 30
GNANI_TTS_TIMEOUT_SECONDS = 40


class PipelineTimeoutError(TimeoutError):
    """Raised when a stage or the overall voice pipeline exceeds its budget."""


def remaining_time(deadline, stage_limit):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise PipelineTimeoutError(
            "The 2-minute processing limit was reached. Please try a shorter recording."
        )
    return min(stage_limit, remaining)


# ---------- Cached Gemini client ----------

@st.cache_resource
def get_gemini_client(api_key, timeout_ms):
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=timeout_ms)
    )


# ---------- Language routing ----------

def choose_tts_language(text, detected):
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))

    if has_devanagari and has_latin:
        return "auto"

    if detected == "hi-IN" and has_latin:
        return "auto"

    supported = {
        "hi-IN", "en-IN", "bn-IN", "gu-IN", "kn-IN",
        "ml-IN", "mr-IN", "pa-IN", "ta-IN", "te-IN"
    }

    if detected in supported:
        return detected
    if has_devanagari:
        return "hi-IN"

    return "en-IN"


# ---------- Prisma speech recognition ----------

def transcribe_audio(audio_bytes, filename, deadline):
    stage_timeout = remaining_time(deadline, GNANI_STT_TIMEOUT_SECONDS)
    response = requests.post(
        STT_URL,
        headers={"X-API-Key-ID": GNANI_API_KEY},
        files={
            "audio_file": (
                filename,
                audio_bytes,
                "application/octet-stream"
            )
        },
        data={
            "model": "prisma-v3.0",
            "language_code": "indic-auto"
        },
        timeout=(min(10, stage_timeout), stage_timeout)
    )
    response.raise_for_status()
    result = response.json()

    transcript = result.get("transcript", "")
    detected = result.get("detected_language_code", "unknown")

    return transcript, detected


# ---------- Gemini intelligence ----------

def generate_answer(user_text, deadline):
    stage_timeout = remaining_time(deadline, GEMINI_TIMEOUT_SECONDS)
    client = get_gemini_client(
        GEMINI_API_KEY, max(1000, int(stage_timeout * 1000))
    )

    # Keep the instruction compact and the spoken answer concise.
    prompt = (
        "You are Bharat Voice AI, a helpful multilingual Indian voice assistant. "
        "Understand the user's intent and answer accurately. "
        "Reply in the same language and style as the user, preserving Hindi, "
        "English, and Hinglish naturally. Keep the answer concise for spoken "
        "delivery (normally 2-4 sentences). Do not invent facts. "
        "If unsure, say so clearly.\n\n"
        f"User message:\n{user_text[:5000]}"
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=180
        )
    )

    answer = (response.text or "").strip()
    if not answer:
        raise ValueError(
            "Gemini returned an empty response. Please try a shorter voice query."
        )

    return answer


# ---------- Timbre speech generation ----------

def generate_speech(text, language, deadline):
    stage_timeout = remaining_time(deadline, GNANI_TTS_TIMEOUT_SECONDS)
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
            "sample_width": 2
        }
    }

    response = requests.post(
        TTS_URL,
        headers={
            "X-API-Key-ID": GNANI_API_KEY,
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=(min(10, stage_timeout), stage_timeout)
    )

    response.raise_for_status()

    if not response.content:
        raise ValueError("Timbre returned an empty audio response.")

    return response.content


# ---------- Interface ----------

st.title("🎙️ Bharat Voice AI")
st.caption("Every Indian Voice. One Intelligent Platform.")

st.markdown(
    "Speak naturally. Bharat Voice AI listens, understands "
    "and responds with a generated voice."
)

if not GNANI_API_KEY or not GEMINI_API_KEY:
    st.error(
        "API keys are missing. Configure GNANI_API_KEY and "
        "GEMINI_API_KEY in Streamlit Secrets."
    )
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

st.subheader("Talk to Bharat Voice AI")

audio_source = st.radio(
    "Choose your audio input",
    ["Record voice", "Upload audio"],
    horizontal=True
)

audio_file = None

if audio_source == "Record voice":
    audio_file = st.audio_input("Tap to record your voice")
else:
    audio_file = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "flac", "aac", "ogg", "m4a"]
    )

if st.button("Process my voice", type="primary", use_container_width=True):
    if audio_file is None:
        st.warning("Please record or upload audio first.")
    else:
        timings = {}
        pipeline_start = time.monotonic()
        deadline = pipeline_start + TOTAL_PIPELINE_TIMEOUT_SECONDS

        try:
            # Stage 1: Speech recognition
            stt_start = time.perf_counter()
            with st.status("1/3 — Bharat Voice AI is listening...", expanded=True) as stt_status:
                audio_bytes = audio_file.getvalue()
                transcript, detected = transcribe_audio(
                    audio_bytes,
                    audio_file.name or "recording.wav",
                    deadline
                )
                timings["Speech recognition"] = time.perf_counter() - stt_start
                stt_status.update(
                    label="1/3 — Speech recognized",
                    state="complete"
                )

            if not transcript.strip():
                st.warning("No speech was detected. Please try again.")
                st.stop()

            st.subheader("You said")
            st.write(transcript)
            st.caption(f"Detected language: {detected}")

            # Stage 2: Gemini response
            gemini_start = time.perf_counter()
            with st.status("2/3 — Gemini is preparing your answer...", expanded=True) as gemini_status:
                try:
                    answer = generate_answer(transcript, deadline)
                    timings["AI response"] = time.perf_counter() - gemini_start
                    gemini_status.update(
                        label="2/3 — AI response ready",
                        state="complete"
                    )
                except Exception:
                    gemini_status.update(
                        label="2/3 — AI response failed or timed out",
                        state="error"
                    )
                    raise

            st.subheader("Bharat Voice AI says")
            st.write(answer)

            # Stage 3: Timbre voice generation
            tts_language = choose_tts_language(transcript, detected)
            tts_start = time.perf_counter()

            with st.status("3/3 — Generating your voice response...", expanded=True) as tts_status:
                try:
                    audio_output = generate_speech(answer, tts_language, deadline)
                    timings["Voice generation"] = time.perf_counter() - tts_start
                    tts_status.update(
                        label="3/3 — Voice response ready",
                        state="complete"
                    )
                except Exception:
                    tts_status.update(
                        label="3/3 — Voice generation failed or timed out",
                        state="error"
                    )
                    raise

            st.subheader("Listen to the response")
            st.audio(audio_output, format="audio/wav")

            st.session_state.history.append({
                "user": transcript,
                "assistant": answer
            })

            st.success("Conversation completed successfully!")

            st.subheader("Performance")
            total_time = time.monotonic() - pipeline_start
            metric_cols = st.columns(3)
            metric_cols[0].metric(
                "Speech recognition",
                f"{timings['Speech recognition']:.1f}s"
            )
            metric_cols[1].metric(
                "AI response",
                f"{timings['AI response']:.1f}s"
            )
            metric_cols[2].metric(
                "Voice generation",
                f"{timings['Voice generation']:.1f}s"
            )
            st.caption(f"Total processing time: {total_time:.1f} seconds")

        except (requests.exceptions.Timeout, PipelineTimeoutError, TimeoutError):
            elapsed = time.monotonic() - pipeline_start
            st.error(
                f"Processing timed out after {elapsed:.1f} seconds. "
                "The maximum processing window is 120 seconds. "
                "Please try a shorter recording and retry."
            )

        except requests.exceptions.HTTPError as error:
            st.error(f"Gnani API error: {error}")
            if error.response is not None:
                st.code(error.response.text[:1500])

        except Exception as error:
            error_text = str(error)
            if "timeout" in error_text.lower() or "deadline" in error_text.lower():
                st.error(
                    "Gemini took too long to respond. Please try again "
                    "with a shorter voice query."
                )
            else:
                st.error(f"Something went wrong: {error}")


# ---------- Conversation history ----------

if st.session_state.history:
    st.divider()
    st.subheader("Conversation history")

    for index, item in enumerate(
        reversed(st.session_state.history), 1
    ):
        with st.expander(f"Conversation {index}"):
            st.markdown("**You:**")
            st.write(item["user"])
            st.markdown("**Bharat Voice AI:**")
            st.write(item["assistant"])

    if st.button("Clear conversation history"):
        st.session_state.history = []
        st.rerun()

st.divider()
st.caption("Bharat Voice AI | Powered by Gnani Prisma, Gemini and Timbre | 120-second processing limit")
