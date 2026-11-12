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

# These are per-request safety timeouts, NOT an overall pipeline deadline.
# They prevent a stalled provider connection from hanging the Streamlit run.
STT_TIMEOUT = (5, 35)
GEMINI_TIMEOUT_MS = 35_000
TTS_TIMEOUT = (5, 40)


@st.cache_resource
def get_gemini_client(api_key):
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
    )


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
    result = response.json()

    transcript = result.get("transcript", "")
    detected = result.get("detected_language_code", "unknown")
    return transcript, detected


# -------------------- Gemini response --------------------

def generate_answer(user_text):
    client = get_gemini_client(GEMINI_API_KEY)

    # Short prompt and response keep generation quick for voice conversations.
    prompt = (
        "You are Bharat Voice AI, a fast, helpful multilingual Indian voice "
        "assistant. Understand the user's intent and answer accurately. "
        "Reply in the same language and style: Hindi, English, or natural "
        "Hinglish. Keep the spoken answer brief, usually 1-3 sentences. "
        "Do not invent facts; say when you are unsure.\n\n"
        f"User: {user_text[:2500]}"
    )

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
        raise ValueError("Gemini returned an empty response. Please try again.")
    return answer


# -------------------- Gnani Timbre TTS --------------------

def generate_speech(text, language):
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

    if not response.content:
        raise ValueError("Timbre returned an empty audio response.")
    return response.content


# -------------------- Streamlit interface --------------------

st.title("🎙️ Bharat Voice AI")
st.caption("Every Indian Voice. One Intelligent Platform.")

st.markdown(
    "Speak naturally. Bharat Voice AI listens, understands, "
    "and responds with a generated voice."
)

if not GNANI_API_KEY or not GEMINI_API_KEY:
    st.error(
        "API keys are missing. Add GNANI_API_KEY and GEMINI_API_KEY "
        "in Streamlit Secrets."
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
                transcript, detected = transcribe_audio(
                    audio_bytes,
                    audio_file.name or "recording.wav",
                )
                timings["STT"] = time.perf_counter() - stage_start
                status.update(label="1/3 — Speech recognized", state="complete")

            if not transcript.strip():
                st.warning("No speech was detected. Please try recording again.")
                st.stop()

            st.subheader("You said")
            st.write(transcript)
            st.caption(f"Detected language: {detected}")

            # Stage 2: Gemini
            with st.status("2/3 — Generating the AI response...", expanded=True) as status:
                stage_start = time.perf_counter()
                answer = generate_answer(transcript)
                timings["Gemini"] = time.perf_counter() - stage_start
                status.update(label="2/3 — AI response ready", state="complete")

            st.subheader("Bharat Voice AI says")
            st.write(answer)

            # Stage 3: TTS
            tts_language = choose_tts_language(transcript, detected)
            with st.status("3/3 — Generating your voice response...", expanded=True) as status:
                stage_start = time.perf_counter()
                audio_output = generate_speech(answer, tts_language)
                timings["TTS"] = time.perf_counter() - stage_start
                status.update(label="3/3 — Voice response ready", state="complete")

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

        except requests.exceptions.Timeout:
            st.error(
                "A provider request took too long and was stopped. "
                "Please retry; a shorter recording may also help."
            )
        except requests.exceptions.HTTPError as error:
            st.error(f"API request failed: {error}")
            if error.response is not None:
                st.code(error.response.text[:1200])
        except Exception as error:
            st.error(f"Something went wrong: {error}")


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
