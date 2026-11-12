
import os
import re
import time
import streamlit as st
import requests
from google import genai

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

def transcribe_audio(audio_bytes, filename):
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
        timeout=120
    )

    response.raise_for_status()
    result = response.json()

    return (
        result.get("transcript", ""),
        result.get("detected_language_code", "unknown")
    )


# ---------- Gemini intelligence ----------

def generate_answer(user_text):
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
You are Bharat Voice AI, a helpful multilingual Indian voice assistant.

Understand the user's intent and answer accurately.
Respond in the same language and style as the user.
Preserve Hindi, English and Hinglish naturally.
Keep the answer concise and suitable for spoken delivery.
Do not invent facts.

User message:
{user_text}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    return response.text.strip()


# ---------- Timbre speech generation ----------

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
        timeout=120
    )

    response.raise_for_status()
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
        try:
            with st.spinner("Bharat Voice AI is listening..."):
                audio_bytes = audio_file.getvalue()
                transcript, detected = transcribe_audio(
                    audio_bytes,
                    audio_file.name or "recording.wav"
                )

            if not transcript.strip():
                st.warning("No speech was detected. Please try again.")
                st.stop()

            st.subheader("You said")
            st.write(transcript)
            st.caption(f"Detected language: {detected}")

            with st.spinner("Thinking..."):
                answer = generate_answer(transcript)

            st.subheader("Bharat Voice AI says")
            st.write(answer)

            tts_language = choose_tts_language(
                transcript, detected
            )

            with st.spinner("Generating your voice response..."):
                audio_output = generate_speech(
                    answer, tts_language
                )

            st.subheader("Listen to the response")
            st.audio(audio_output, format="audio/wav")

            st.session_state.history.append({
                "user": transcript,
                "assistant": answer
            })

            st.success("Conversation completed successfully!")

        except requests.exceptions.HTTPError as error:
            st.error(f"Gnani API error: {error}")
            if error.response is not None:
                st.code(error.response.text[:1500])

        except Exception as error:
            st.error(f"Something went wrong: {error}")

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
st.caption("Bharat Voice AI | Powered by Gnani Prisma, Gemini and Timbre")
