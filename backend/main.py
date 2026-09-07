"""
FastAPI Backend - AI Voice Cloning + Synthetic Voice Detection
Main application entry point.
"""
import os
import uuid
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.config import (
    HOST, PORT, UPLOAD_DIR, GENERATED_DIR, MAX_UPLOAD_SIZE_MB,
    ALLOWED_AUDIO_EXTENSIONS, DEMO_MODE
)
from backend.voice_cloner import cloner
from backend.voice_detector import detector

# ─── App Setup ──────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Voice Cloning + Detection",
    description="SIH Project: Voice cloning with consent and synthetic voice detection",
    version="1.0.0"
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ─── In-memory storage (simple alternative to SQLite for prototype) ────
activity_log = []


def log_activity(action: str, details: str, status: str = "success"):
    """Log an activity for the dashboard."""
    activity_log.append({
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "details": details,
        "status": status,
    })
    # Keep only last 100 entries
    if len(activity_log) > 100:
        activity_log.pop(0)


# ─── Helper ────────────────────────────────────────────────────────────
def validate_audio_file(file: UploadFile) -> str:
    """Validate uploaded audio file and return the extension."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
        )
    return ext


async def save_upload(file: UploadFile, prefix: str = "upload") -> str:
    """Save an uploaded file and return the path."""
    ext = validate_audio_file(file)
    filename = f"{prefix}_{uuid.uuid4().hex[:12]}{ext}"
    filepath = str(UPLOAD_DIR / filename)

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_UPLOAD_SIZE_MB}MB"
        )

    with open(filepath, "wb") as f:
        f.write(content)

    return filepath


# ─── Routes ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Frontend not found. Check frontend/index.html</h1>")


@app.get("/api/status")
async def get_status():
    """Get system status including model availability."""
    cloner_status = cloner.get_status()
    return {
        "status": "online",
        "ready": detector.use_ml,
        "demo_mode": cloner_status["demo_mode"],
        "cloner": cloner_status,
        "detector": {
            "status": "ready" if detector.use_ml else "heuristic",
            "method": "ml_random_forest" if detector.use_ml else "spectral_analysis",
            "model": detector.model_name if detector.use_ml else "none",
        },
        "version": "1.0.0",
    }


# ─── Voice Cloning Endpoints ──────────────────────────────────────────

@app.post("/api/clone")
async def clone_voice(
    reference_audio: UploadFile = File(...),
    text: str = Form(...),
    consent: str = Form(...),
    language: str = Form("en"),
):
    """
    Clone a voice: upload reference audio + text -> get generated speech.
    Requires consent checkbox to be checked.
    """
    # Validate consent
    if consent.lower() not in ("true", "yes", "1", "on"):
        raise HTTPException(
            status_code=400,
            detail="Consent is required. Please confirm you have authorization to use this voice."
        )

    # Validate text
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="Text too long. Maximum 2000 characters.")

    # Save reference audio
    try:
        ref_path = await save_upload(reference_audio, prefix="ref")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save reference audio: {str(e)}")

    # Generate cloned voice
    try:
        start_time = time.time()
        result = cloner.clone_voice(text, ref_path, language)
        elapsed = round(time.time() - start_time, 2)

        log_activity(
            "voice_clone",
            f"Text: '{text[:50]}...' | Mode: {result['mode']} | Time: {elapsed}s",
            "success"
        )

        return {
            "success": True,
            "audio_url": f"/api/audio/{result['output_filename']}",
            "download_url": f"/api/download/{result['output_filename']}",
            "duration": result["duration"],
            "mode": result["mode"],
            "model": result["model"],
            "processing_time": elapsed,
            "disclaimer": result["disclaimer"],
        }

    except Exception as e:
        log_activity("voice_clone", f"Error: {str(e)}", "error")
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")


# ─── Voice Detection Endpoints ────────────────────────────────────────

@app.post("/api/detect")
async def detect_voice(
    file: UploadFile = File(...),
):
    """
    Analyze an audio file to determine if it's AI-generated or human.
    Returns probabilistic results with confidence scores.
    """
    # Save uploaded audio
    try:
        filepath = await save_upload(file, prefix="detect")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save audio: {str(e)}")

    # Analyze
    try:
        start_time = time.time()
        result = detector.analyze(filepath)
        elapsed = round(time.time() - start_time, 2)

        log_activity(
            "voice_detection",
            f"Result: {result['prediction']} ({result['confidence']}%) | Time: {elapsed}s",
            "success"
        )

        return {
            "success": True,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "ai_probability": result.get("ai_probability", 50),
            "risk_level": result.get("risk_level", "UNKNOWN"),
            "features": result.get("features", {}),
            "scores": result.get("scores", {}),
            "details": result.get("details", []),
            "explanation": result.get("details", []),
            "processing_time": elapsed,
            "disclaimer": result["disclaimer"],
        }

    except Exception as e:
        log_activity("voice_detection", f"Error: {str(e)}", "error")
        raise HTTPException(status_code=500, detail=f"Voice detection failed: {str(e)}")


# ─── Audio Serving ────────────────────────────────────────────────────

@app.get("/api/audio/{filename}")
async def get_audio(filename: str):
    """Serve generated audio files for playback."""
    filepath = GENERATED_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(
        str(filepath),
        media_type="audio/wav",
        headers={"Accept-Ranges": "bytes"}
    )


@app.get("/api/download/{filename}")
async def download_audio(filename: str):
    """Download generated audio files."""
    filepath = GENERATED_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(
        str(filepath),
        media_type="audio/wav",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ─── Dashboard ────────────────────────────────────────────────────────

@app.get("/api/dashboard")
async def get_dashboard():
    """Get dashboard data including activity log and statistics."""
    clone_count = sum(1 for a in activity_log if a["action"] == "voice_clone" and a["status"] == "success")
    detect_count = sum(1 for a in activity_log if a["action"] == "voice_detection" and a["status"] == "success")
    error_count = sum(1 for a in activity_log if a["status"] == "error")

    return {
        "statistics": {
            "total_clones": clone_count,
            "total_detections": detect_count,
            "total_errors": error_count,
            "total_activities": len(activity_log),
        },
        "recent_activity": list(reversed(activity_log[-20:])),
        "system": cloner.get_status(),
    }


# ─── Startup ──────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    print("\n" + "=" * 60)
    print("  AI Voice Cloning + Synthetic Voice Detection")
    print("  SIH Project Prototype")
    print("=" * 60)
    print(f"  Server:    http://{HOST}:{PORT}")
    print(f"  Demo Mode: {cloner.get_status()['demo_mode']}")
    print(f"  Detector:  Ready (Spectral Analysis)")
    print("=" * 60 + "\n")


# ─── Run ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info",
    )
