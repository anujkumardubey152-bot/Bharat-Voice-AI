# 🇮🇳 Bharat Voice AI
### Every Indian Voice. One Intelligent Platform.

**A multilingual, voice-first AI assistant that listens, understands, and responds naturally using AI-powered speech recognition, conversational intelligence, and voice synthesis.**

<p align="center">
  <a href="https://bharat-voice.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_LIVE_DEMO-Try_Bharat_Voice_AI-FF4B4B?style=for-the-badge" alt="Live Demo"/>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=flat-square&logo=streamlit"/>
  <img src="https://img.shields.io/badge/Speech-Gnani%20Prisma-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/AI-Google%20Gemini-4285F4?style=flat-square&logo=google"/>
  <img src="https://img.shields.io/badge/Voice-Gnani%20Timbre-purple?style=flat-square"/>
</p>

---

## 🚀 Live Application

### [Try Bharat Voice AI →](https://bharat-voice.streamlit.app/)

Experience natural voice interaction:
- Record your voice directly.
- Upload an audio file.
- Get an AI-generated response.
- Listen to the response in generated speech.

---

## 💡 About Bharat Voice AI

India's linguistic diversity deserves AI that understands how people actually communicate.

Bharat Voice AI brings together speech recognition, generative artificial intelligence, and speech synthesis into one unified voice-based experience.

It enables users to communicate naturally through spoken language, including Hindi, English, and Hinglish.

The platform transforms voice input into meaningful AI-generated conversations and converts the responses back into speech.

**Our vision:** Make intelligent AI interaction more accessible, natural, and inclusive for people across India.

## ✨ Key Features

| Feature | Description |
|---|---|
| 🎙️ Voice Recording | Record audio directly through the application. |
| 📂 Audio Upload | Upload supported audio files for processing. |
| 🗣️ Speech-to-Text | Convert spoken language into text using Gnani Prisma. |
| 🌐 Language Detection | Automatically identify the detected speech language. |
| 🧠 AI Intelligence | Generate contextual responses using Google Gemini. |
| 🔊 Text-to-Speech | Convert AI-generated responses into natural audio using Gnani Timbre. |
| 🇮🇳 Multilingual Interaction | Handle Hindi, English, and Hinglish conversations. |
| 💬 Conversation History | Review previous interactions within the session. |
| ⚡ Performance Tracking | Measure processing time across voice pipeline stages. |

## 🏗️ System Architecture

Bharat Voice AI uses a sequential AI-powered voice processing pipeline.

```text
                USER
                 |
                 v
        VOICE INPUT / UPLOAD
                 |
                 v
          GNANI PRISMA
          Speech-to-Text
                 |
                 v
       TRANSCRIPT + LANGUAGE
                 |
                 v
           GOOGLE GEMINI
        AI Reasoning & Response
                 |
                 v
          GENERATED TEXT
                 |
                 v
          GNANI TIMBRE
           Text-to-Speech
                 |
                 v
        AI-GENERATED AUDIO
                 |
                 v
             USER
```

### How the System Works

**1. Voice Input**

The user records their voice or uploads an audio file through the Streamlit interface.

**2. Speech Recognition**

Gnani Prisma processes the audio and converts spoken language into text.

**3. Language Detection**

The application uses the detected language to determine the appropriate response configuration.

**4. AI Intelligence**

Google Gemini processes the transcript, understands the user's intent, and generates a contextual response.

**5. Voice Generation**

Gnani Timbre converts the AI-generated text into speech.

**6. Voice Output**

The generated audio is played directly in the application, completing the voice interaction.

## 🛠️ Technology Stack

| Technology | Application |
|---|---|
| Python | Core programming language |
| Streamlit | Web application and interface |
| Gnani Prisma v3.0 | Speech recognition |
| Google Gemini | Conversational intelligence |
| Gnani Timbre v3.0 | Speech synthesis |
| Requests | API communication |
| Google Gen AI SDK | Gemini integration |
| GitHub | Version control |
| Streamlit Community Cloud | Cloud deployment |

## 🧠 Core AI Components

### Gnani Prisma — Speech Recognition
- Processes uploaded and recorded audio.
- Generates text transcripts.
- Supports automatic Indian language detection.
- Enables multilingual speech interaction.

### Google Gemini — Conversational Intelligence
- Understands user input.
- Generates contextual responses.
- Supports natural multilingual conversations.
- Produces concise responses suitable for voice interaction.

### Gnani Timbre — Voice Synthesis
- Converts generated text into speech.
- Supports Indian language voice generation.
- Produces playable audio responses.

## 📂 Project Structure

```text
Bharat-Voice-AI/
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

## ⚙️ Run Locally

### Prerequisites
- Python 3.10 or higher
- Gnani AI API key
- Google Gemini API key

### Step 1: Clone the repository

```bash
git clone https://github.com/anujkumardubey152-bot/Bharat-Voice-AI.git

cd Bharat-Voice-AI
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure API credentials

Create `.streamlit/secrets.toml`:

```toml
GNANI_API_KEY = "your_gnani_api_key"
GEMINI_API_KEY = "your_gemini_api_key"
```

Keep your API credentials private. Never commit secrets to GitHub.

### Step 5: Launch the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 🔐 Security

- API credentials are managed through Streamlit Secrets or environment variables.
- Sensitive credentials are not included in application source code.
- API requests use authenticated service connections.
- Users should never share their private API keys.

## 🎯 Real-World Applications

Bharat Voice AI can support a range of voice-first experiences:

- **Education:** Voice-based learning and conversational assistance.
- **Accessibility:** Easier interaction for users who prefer speaking over typing.
- **Customer Support:** AI-powered multilingual conversational assistance.
- **Digital Inclusion:** More natural access to AI for Indian language speakers.
- **Personal Assistance:** Everyday questions and voice-based information access.

## 👨‍💻 Developer

**Anuj Kumar Dubey**

B.Tech Computer Science and Engineering  
Institute of Engineering and Rural Technology (IERT), Prayagraj

**Technical Interests**
- Artificial Intelligence and Machine Learning
- Generative AI and Large Language Models
- AI Agents and Automation
- Applied Machine Learning
- Intelligent Voice Applications

**GitHub:** [@anujkumardubey152-bot](https://github.com/anujkumardubey152-bot)

**Live Project:** [Bharat Voice AI](https://bharat-voice.streamlit.app/)

## 🙏 Acknowledgements

Special thanks to:

- [Gnani AI](https://gnani.ai/) — Speech recognition and voice synthesis.
- [Google AI](https://ai.google.dev/) — Gemini conversational intelligence.
- [Streamlit](https://streamlit.io/) — Application development and deployment.

## 📜 License

This project is available for educational, demonstration, and research purposes. All third-party APIs, models, and services remain subject to their respective terms and licenses.

---

<p align="center">
  <b>🇮🇳 Bharat Voice AI</b><br/>
  Every Indian Voice. One Intelligent Platform.
  <br/><br/>
  <a href="https://bharat-voice.streamlit.app/">Experience the Live Application</a>
</p>
