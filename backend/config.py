"""
Configuration for the Voice Cloning + Detection application.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
UPLOAD_DIR = BASE_DIR / "uploads"
GENERATED_DIR = BASE_DIR / "generated"
MODELS_DIR = BASE_DIR / "models"
STORAGE_DIR = BASE_DIR / "storage"
ENROLLED_VOICES_DIR = STORAGE_DIR / "enrolled_voices"

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
GENERATED_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
STORAGE_DIR.mkdir(exist_ok=True)
ENROLLED_VOICES_DIR.mkdir(exist_ok=True)

# Server settings
HOST = "0.0.0.0"
PORT = 8000

# Audio settings
MAX_UPLOAD_SIZE_MB = 50
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm", ".aac", ".mp4", ".3gp"}
SAMPLE_RATE = 22050

# Demo mode - set to True when real ML models can't run
# Real TTS models (XTTS, OpenVoice) require CUDA GPU + specific Python versions
DEMO_MODE = True  # Will be auto-detected on startup

# Detection thresholds
DETECTION_CONFIDENCE_THRESHOLD = 0.5
