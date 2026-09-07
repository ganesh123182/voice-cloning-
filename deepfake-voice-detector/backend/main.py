"""
Deepfake Voice Detector - FastAPI Backend
Main application entry point.

Run with:
    python -m backend.main
"""
import os
import sys
import uuid
import time
import signal
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    HOST, PORT, UPLOAD_DIR, OUTPUT_DIR,
    MAX_UPLOAD_MB, ALLOWED_EXTENSIONS,
)

# ---- App Setup ----
app = FastAPI(
    title="Deepfake Voice Detector",
    description="SIH Project - AI-powered deepfake voice detection using pretrained Wav2Vec2 model",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (frontend)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ---- Activity Log (in-memory) ----
activity_log = []

def log_activity(action: str, details: str, status: str = "success"):
    activity_log.append({
        "id": uuid.uuid4().hex[:8],
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "details": details,
        "status": status,
    })
    if len(activity_log) > 200:
        activity_log.pop(0)


# ---- Lazy model loading ----
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        from .detector import get_detector as _get
        _detector = _get()
    return _detector


# ---- Helpers ----
def validate_audio(file: UploadFile) -> str:
    if not file.filename:
        raise HTTPException(400, "No filename")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    return ext

async def save_upload(file: UploadFile) -> str:
    ext = validate_audio(file)
    filename = f"upload_{uuid.uuid4().hex[:12]}{ext}"
    filepath = UPLOAD_DIR / filename
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {MAX_UPLOAD_MB}MB)")
    with open(filepath, "wb") as f:
        f.write(content)
    return str(filepath)


# ---- Routes ----

@app.get("/", response_class=HTMLResponse)
async def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend not found</h1>")


@app.get("/api/status")
def status():
    det = get_detector()
    info = det.get_status()
    return {
        "status": "online",
        "detector": info,
        "version": "1.0.0",
    }


@app.post("/api/detect")
async def detect_deepfake(audio_file: UploadFile = File(...)):
    """
    Upload an audio file and detect if it's a deepfake.
    Returns prediction, confidence, risk level, and explanation.
    """
    # Save file
    try:
        filepath = await save_upload(audio_file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {e}")

    # Detect
    import asyncio
    try:
        det = get_detector()
        result = await asyncio.to_thread(det.detect, filepath)

        log_activity(
            "detection",
            f"{result['prediction']} ({result['confidence']}%) | {result['risk_level']} | {result['processing_time']}s",
            "success" if result["prediction"] != "ERROR" else "error",
        )

        return {
            "success": True,
            "filename": audio_file.filename,
            **result,
        }

    except Exception as e:
        log_activity("detection", f"Error: {e}", "error")
        raise HTTPException(500, f"Detection failed: {e}")


@app.get("/api/history")
async def get_history():
    """Get recent detection activity."""
    return {
        "total": len(activity_log),
        "recent": list(reversed(activity_log[-30:])),
    }


# ---- Serve uploaded audio for playback ----
@app.get("/api/audio/{filename}")
async def serve_audio(filename: str):
    # Check both uploads and outputs
    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        fp = directory / filename
        if fp.exists():
            return FileResponse(str(fp), media_type="audio/wav")
    raise HTTPException(404, "File not found")


# ---- Kill stale process on same port (Windows) ----
def _free_port(port: int):
    """Kill any leftover process occupying our port so restart works cleanly."""
    if sys.platform != "win32":
        return
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1]
                # Don't kill ourselves
                if pid != str(os.getpid()):
                    subprocess.run(["taskkill", "/F", "/PID", pid],
                                   capture_output=True, timeout=5)
                    print(f"[INFO] Killed stale process PID {pid} on port {port}")
    except Exception:
        pass  # Best effort


# ---- Startup ----
@app.on_event("startup")
async def startup():
    print()
    print("=" * 60)
    print("  DEEPFAKE VOICE DETECTOR")
    print("  SIH Project Prototype")
    print("=" * 60)
    print(f"  URL:  http://{HOST}:{PORT}")
    print("  Loading model on first request...")
    print("=" * 60)
    print()


# ---- Shutdown ----
@app.on_event("shutdown")
async def shutdown():
    """Clean up resources on shutdown to avoid stale locks on restart."""
    global _detector
    print("[INFO] Shutting down, cleaning up resources...")
    _detector = None
    # Clean up temp upload files
    try:
        for f in UPLOAD_DIR.glob("upload_*"):
            try:
                f.unlink()
            except Exception:
                pass
    except Exception:
        pass
    print("[INFO] Cleanup complete.")


# ---- Main ----
if __name__ == "__main__":
    import uvicorn
    # Free the port in case a previous instance is still lingering
    _free_port(PORT)
    uvicorn.run(
        "backend.main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )
