"""
Deepfake Voice Detector - Configuration
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
UPLOAD_DIR = PROJECT_DIR / "uploads"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

HOST = "127.0.0.1"
PORT = 8001

MAX_UPLOAD_MB = 50
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}
TARGET_SAMPLE_RATE = 16000

# Model settings
MODEL_ID = "garystafford/wav2vec2-deepfake-voice-detector"
DEVICE = "cpu"  # Will be updated on startup if CUDA available
