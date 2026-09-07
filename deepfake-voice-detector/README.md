# Deepfake Voice Detector (SIH Prototype)

A fast, fully functional prototype for deepfake voice detection, designed for the Smart India Hackathon.
This application uses a pretrained HuggingFace `wav2vec2` model to analyze audio files and detect synthetic or AI-generated speech.

## Features
- **Real AI Detection**: Integrates `garystafford/wav2vec2-deepfake-voice-detector` via HuggingFace Transformers.
- **Demo Fallback**: Automatically falls back to a spectral heuristics analysis if the ML model cannot be loaded, so the UI is always testable.
- **Professional UI**: Modern, dark-themed, glassmorphic UI built with vanilla HTML/CSS/JS.
- **FastAPI Backend**: Robust, asynchronous backend.

## Requirements
- Python 3.10+
- RAM: 8GB minimum recommended
- GPU: Optional (runs on CPU, faster with CUDA)

## Quick Start (Windows)
1. Double-click `start.bat`.
2. The script will automatically install dependencies and start the server.
3. Open `http://127.0.0.1:8000` in your browser.

## Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python -m backend.main
```

## How to Test
1. Open the UI.
2. Upload a known real human voice recording (e.g., record yourself on your phone).
3. Click "Analyze Voice" and observe the "REAL / HUMAN" prediction.
4. Upload an AI-generated voice (e.g., from ElevenLabs, TTS tools).
5. Click "Analyze Voice" and observe the "AI GENERATED" prediction and confidence score.

**Note**: First-time analysis may take a few moments as the model is downloaded and loaded into memory.
