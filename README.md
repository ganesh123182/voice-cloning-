# 🎙️ VoiceForge AI — Voice Cloning + Synthetic Voice Detection

**SIH Project Prototype** — AI-powered voice cloning with consent management and synthetic voice detection.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn numpy python-multipart
```

**Optional** (for non-WAV file support):
```bash
pip install pydub
```

### 2. Run the Application

```bash
cd "path/to/voice cloning"
python -m backend.main
```

### 3. Open in Browser

Navigate to: **http://127.0.0.1:8000**

---

## 📋 Features

### 🎤 Voice Cloning (Demo Mode)
1. Go to the **Voice Clone** tab
2. Upload a reference voice audio file (WAV, MP3, OGG, FLAC)
3. Enter the text you want spoken
4. Check the **consent checkbox** (required)
5. Click **Generate Voice**
6. Play the generated audio and download as WAV

> **Note:** Voice cloning runs in **Demo Mode** — it generates synthetic audio using basic signal processing. Real AI voice cloning (XTTS/OpenVoice) requires a CUDA-enabled GPU with compatible Python version. The UI workflow is fully functional.

### 🔍 Voice Analyzer (Real Detection)
1. Go to the **Voice Analyzer** tab
2. Upload any speech audio file
3. Click **Analyze Audio**
4. View the results:
   - **Prediction**: Likely Human / Likely AI-Generated / Inconclusive
   - **Confidence score** (percentage)
   - **Feature breakdown** with spectral analysis details
   - **Disclaimer** (results are probabilistic)

> The voice detection uses **real spectral analysis** — it analyzes spectral flatness, energy variation, pitch patterns, zero-crossing rates, and bandwidth to classify audio. Results are clearly labelled as probabilistic.

### 📊 Dashboard
- View total clones, detections, and errors
- System status and model information
- Recent activity log with timestamps

---

## 🏗️ Project Structure

```
voice cloning/
├── backend/
│   ├── __init__.py
│   ├── main.py            # FastAPI application entry point
│   ├── config.py           # Configuration settings
│   ├── audio_utils.py      # Audio processing utilities
│   ├── voice_cloner.py     # Voice cloning module (demo + real)
│   ├── voice_detector.py   # Voice detection module (spectral analysis)
│   ├── uploads/            # Uploaded audio files
│   └── generated/          # Generated audio files
├── frontend/
│   ├── index.html          # Main HTML page
│   ├── styles.css          # Styles (glassmorphism dark theme)
│   └── app.js              # Frontend application logic
└── README.md
```

---

## 🔧 Technology Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python, FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript (vanilla) |
| Audio Processing | NumPy, Wave (stdlib) |
| Voice Detection | Spectral Analysis (FFT-based heuristics) |
| Voice Cloning | Demo Synthesizer (real models need CUDA) |

---

## 🧪 Testing

### Test Voice Cloning
1. Prepare any short audio file (WAV recommended, 5-30 seconds)
2. Upload it as the reference voice
3. Type any text (e.g., "Hello, this is a test of the voice cloning system")
4. Check consent and generate
5. Play the result — you'll hear a synthetic tone (Demo Mode)

### Test Voice Detection
1. Upload any speech audio file
2. Click Analyze
3. Check the prediction, confidence score, and feature breakdown
4. Try with different audio types:
   - A recording of your own voice → should lean toward "Likely Human"
   - AI-generated speech (e.g., from text-to-speech tools) → may detect AI patterns

### API Testing (curl)

```bash
# Check status
curl http://127.0.0.1:8000/api/status

# Voice detection
curl -X POST http://127.0.0.1:8000/api/detect -F "audio_file=@your_audio.wav"

# Voice cloning
curl -X POST http://127.0.0.1:8000/api/clone \
  -F "reference_audio=@reference.wav" \
  -F "text=Hello world" \
  -F "consent=true"

# Dashboard
curl http://127.0.0.1:8000/api/dashboard
```

---

## ⚠️ Important Notes

- **Demo Mode**: Voice cloning uses a simple synthesizer, not real AI. Real models (XTTS v2, OpenVoice) require CUDA GPU + Python 3.10/3.11 + TTS library.
- **Detection is probabilistic**: The voice analyzer uses spectral heuristics. It should NOT be used as definitive proof. No detection system is 100% accurate.
- **Consent required**: Voice cloning requires explicit consent checkbox. Only authorized/demo voices should be used.
- **File support**: WAV files work out of the box. For MP3/OGG/FLAC, install `pydub` and `ffmpeg`.

---

## 📄 License

SIH Project — Educational / Research Use
