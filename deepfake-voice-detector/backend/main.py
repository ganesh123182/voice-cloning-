"""
Deepfake Voice Detector - FastAPI Backend
Main application entry point.

Run with:
    python -m backend.main
"""
from asyncio import subprocess
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from sqlalchemy.orm import Session
import bcrypt
from jose import JWTError, jwt
from datetime import timedelta

from .config import (
    HOST, PORT, UPLOAD_DIR, OUTPUT_DIR,
    MAX_UPLOAD_MB, ALLOWED_EXTENSIONS,
)
from .database import engine, Base, get_db
from .models import User, VoiceProfile, MonitoringSession

# Initialize Database
Base.metadata.create_all(bind=engine)

# ---- Trust Voice Auth Settings ----
SECRET_KEY = "trustvoice-super-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception:
        return False

def get_password_hash(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

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

# ---- Trust Voice Extended APIs ----

@app.post("/api/auth/register")
def register_user(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(password)
    user = User(email=email, hashed_password=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "User registered successfully", "id": user.id}

@app.post("/api/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/auth/me")
def read_users_me(current_user: User = Depends(get_current_user)):
    return {"id": current_user.id, "email": current_user.email}

@app.post("/api/voice/enrollment")
async def voice_enrollment(
    file: UploadFile = File(...), 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Receive a voice sample, extract features, and register voice profile."""
    import librosa
    import soundfile as sf
    from .ai_pipeline import extract_embedding
    import torch
    import json
    import hashlib

    # Check if user already has a profile
    existing = db.query(VoiceProfile).filter(VoiceProfile.user_id == current_user.id).first()
    if existing and existing.embedding:
        return {"message": "Voice profile already exists", "profile_hash": existing.profile_hash}
    
    # Save the file
    try:
        filepath = await save_upload(file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {e}")
    
    try:
        # Resample audio to 16kHz mono
        y, sr = librosa.load(filepath, sr=16000, mono=True)
        audio_tensor = torch.from_numpy(y)
        
        # Extract embedding via ai_pipeline
        emb = extract_embedding(audio_tensor)
        emb_list = emb.tolist()
        emb_json = json.dumps(emb_list)
        
        # Generate SHA-256 fingerprint via blockchain module
        from .blockchain import generate_voice_fingerprint, register_voice_profile as bc_register
        profile_hash = generate_voice_fingerprint(emb_list)
        
        # Register on blockchain and get transaction hash
        tx_hash = bc_register(current_user.id, profile_hash)
        
        if existing:
            profile = existing
            profile.embedding = emb_json
            profile.profile_hash = profile_hash
            profile.blockchain_tx_id = tx_hash
            profile.quality_score = 98.5
        else:
            profile = VoiceProfile(
                user_id=current_user.id,
                embedding=emb_json,
                profile_hash=profile_hash,
                blockchain_tx_id=tx_hash,
                quality_score=98.5
            )
            db.add(profile)
            
        db.commit()
        db.refresh(profile)
        return {
            "message": "Enrolled successfully",
            "profile_hash": profile.profile_hash,
            "blockchain_tx_id": tx_hash
        }
    except Exception as e:
        raise HTTPException(500, f"Enrollment processing failed: {e}")

@app.get("/api/voice/profile")
def get_voice_profile(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    profile = db.query(VoiceProfile).filter(VoiceProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {
        "id": profile.id,
        "profile_hash": profile.profile_hash,
        "blockchain_tx_id": profile.blockchain_tx_id,
        "quality_score": profile.quality_score,
        "status": profile.status,
        "created_at": profile.created_at
    }


# ---- Blockchain Audit & Verification ----

@app.get("/api/blockchain/audit/{session_id}")
def get_blockchain_audit(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Return the immutable blockchain audit trail for a monitoring session.
    This endpoint is designed for SIH judges to inspect tamper-proof records.
    """
    from .blockchain import get_session_audit, verify_voice_profile as bc_verify

    session = db.query(MonitoringSession).filter(MonitoringSession.id == session_id).first()
    if not session:
        raise HTTPException(404, "Session not found")

    # Only the session owner can view the audit
    if session.user_id != current_user.id:
        raise HTTPException(403, "Not authorized to view this session")

    profile = db.query(VoiceProfile).filter(VoiceProfile.user_id == current_user.id).first()

    session_data = {
        "user_id": current_user.id,
        "user_email": current_user.email,
        "call_type": session.call_type,
        "audio_mode": session.audio_mode,
        "started_at": session.started_at.isoformat() if session.started_at else None,
        "ended_at": session.ended_at.isoformat() if session.ended_at else None,
        "result": session.result,
        "risk_score": session.risk_score,
        "voice_profile_hash": profile.profile_hash if profile else None,
        "blockchain_tx_id": profile.blockchain_tx_id if profile else None,
    }

    # Verify current voice profile integrity against blockchain
    if profile and profile.profile_hash:
        session_data["blockchain_verified"] = bc_verify(current_user.id, profile.profile_hash)
    else:
        session_data["blockchain_verified"] = False

    audit_report = get_session_audit(session_id, session_data)
    return audit_report


@app.get("/api/blockchain/verify")
def verify_blockchain_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Verify the current user's voice profile against the blockchain ledger."""
    from .blockchain import verify_voice_profile as bc_verify, get_on_chain_record

    profile = db.query(VoiceProfile).filter(VoiceProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(404, "No voice profile found")

    is_valid = bc_verify(current_user.id, profile.profile_hash)
    chain_record = get_on_chain_record(current_user.id)

    return {
        "user_id": current_user.id,
        "profile_hash": profile.profile_hash,
        "blockchain_verified": is_valid,
        "blockchain_tx_id": profile.blockchain_tx_id,
        "on_chain_record": chain_record,
    }

@app.post("/api/monitoring/sessions")
async def start_monitoring(
    audio_file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Simulate a call detection event being logged by Trust Voice app."""
    # Ensure they have a voice profile
    profile = db.query(VoiceProfile).filter(VoiceProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(status_code=400, detail="Voice profile required for monitoring")
    
    # Analyze incoming audio
    try:
        filepath = await save_upload(audio_file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {e}")
        
    import asyncio
    try:
        det = get_detector()
        result = await asyncio.to_thread(det.detect, filepath)
    except Exception as e:
        raise HTTPException(500, f"Detection failed: {e}")
    
    risk_score = result.get("ai_probability", 0.0)
        
    session = MonitoringSession(
        user_id=current_user.id,
        result=result["prediction"],
        risk_score=risk_score
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    
    log_activity(
        "trust_voice_monitoring",
        f"User: {current_user.email} | Result: {result['prediction']} ({result['confidence']}%) | Risk: {risk_score}",
        "success" if result["prediction"] != "ERROR" else "error",
    )
    
    return {
        "message": "Monitoring session logged",
        "session_id": session.id,
        "risk_score": risk_score,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "details": result.get("details", [])
    }

@app.websocket("/api/monitoring/live")
async def live_monitoring(websocket: WebSocket, token: str, db: Session = Depends(get_db)):
    await websocket.accept()
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            await websocket.close(code=1008)
            return
        user = db.query(User).filter(User.email == email).first()
        if not user:
            await websocket.close(code=1008)
            return
    except JWTError:
        await websocket.close(code=1008)
        return

    profile = db.query(VoiceProfile).filter(VoiceProfile.user_id == user.id).first()
    if not profile or not profile.embedding:
        await websocket.send_json({"error": "Voice profile required. Please enroll first."})
        await websocket.close(code=1008)
        return

    import json
    import numpy as np
    import torch
    from .ai_pipeline import is_speech, extract_embedding, is_user_speaking, DeepfakeInferenceEngine, RollingRiskScore
    
    try:
        enrolled_embedding = np.array(json.loads(profile.embedding), dtype=np.float32)
    except Exception:
        await websocket.send_json({"error": "Invalid voice profile data."})
        await websocket.close(code=1008)
        return

    df_engine = DeepfakeInferenceEngine()
    rolling_score = RollingRiskScore()

    session = MonitoringSession(user_id=user.id, call_type="live_websocket", audio_mode="microphone")
    db.add(session)
    db.commit()
    db.refresh(session)
    
    highest_risk = 0.0
    
    try:
        while True:
            # Receive continuous raw 16-bit 16kHz PCM byte chunks
            data = await websocket.receive_bytes()
            if not data:
                continue
                
            # Convert raw bytes to float32 numpy array, normalize from 16-bit PCM
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_data)
            
            # Run Silero VAD
            if not is_speech(audio_tensor):
                await websocket.send_json({"speaker": "silence", "risk_score": 0.0})
                continue
                
            # Speaker Biometrics
            chunk_embedding = extract_embedding(audio_tensor)
            if is_user_speaking(chunk_embedding, enrolled_embedding):
                await websocket.send_json({
                    "speaker": "user",
                    "risk_score": 0.0,
                    "label": "User Speaking"
                })
                continue
                
            # If caller is speaking -> run Deepfake detector
            df_result = df_engine.predict_chunk(audio_tensor)
            
            # Apply rolling risk score
            smoothed_score = rolling_score.update(df_result["fake_prob"]) * 100.0
            highest_risk = max(highest_risk, smoothed_score)
            
            is_alert = smoothed_score > 65.0
            label = "AI Clone" if is_alert else "Human Voice"
            suggestion = "Disconnect immediately" if is_alert else "Safe to proceed"
            
            await websocket.send_json({
                "speaker": "caller",
                "risk_score": round(smoothed_score, 1),
                "label": label,
                "is_alert": is_alert,
                "suggestion": suggestion
            })
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception as e:
        print(f"Error in live monitoring: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        session.ended_at = datetime.utcnow()
        session.risk_score = highest_risk
        db.commit()


@app.websocket("/api/monitoring/simulate")
async def simulate_call_monitoring(websocket: WebSocket, token: str, db: Session = Depends(get_db)):
    """Receives a complete audio file, chunks it server-side, and streams analysis results."""
    await websocket.accept()
    
    # Auth
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            await websocket.close(code=1008)
            return
        user = db.query(User).filter(User.email == email).first()
        if not user:
            await websocket.close(code=1008)
            return
    except JWTError:
        await websocket.close(code=1008)
        return

    profile = db.query(VoiceProfile).filter(VoiceProfile.user_id == user.id).first()
    if not profile:
        await websocket.send_json({"error": "Voice profile required. Please enroll first."})
        await websocket.close(code=1008)
        return

    session = MonitoringSession(user_id=user.id, call_type="simulated_call", audio_mode="file_stream")
    db.add(session)
    db.commit()
    db.refresh(session)

    import asyncio
    import soundfile as sf
    import numpy as np
    highest_risk = 0.0
    upload_path = UPLOAD_DIR / f"sim_{session.id}.bin"

    try:
        # Receive the full file as binary
        file_data = await websocket.receive_bytes()
        if not file_data:
            await websocket.send_json({"error": "No audio data received"})
            return

        with open(upload_path, "wb") as f:
            f.write(file_data)

        await websocket.send_json({
            "status": "processing",
            "message": "File received. Analyzing in chunks...",
            "session_id": session.id
        })

        # Load audio server-side using librosa
        import librosa
        try:
            y, sr = librosa.load(str(upload_path), sr=16000, mono=True)
        except Exception:
            import av
            container = av.open(str(upload_path))
            stream = container.streams.audio[0]
            orig_sr = stream.rate
            samples = []
            for frame in container.decode(stream):
                arr = frame.to_ndarray()
                if arr.shape[0] > 1:
                    arr = np.mean(arr, axis=0)
                else:
                    arr = arr[0]
                samples.append(arr)
            container.close()
            y = np.concatenate(samples).astype(np.float32)
            if orig_sr != 16000:
                y = librosa.resample(y, orig_sr=orig_sr, target_sr=16000)
            sr = 16000

        # Split into 4-second chunks
        chunk_duration = 4
        chunk_samples = chunk_duration * sr
        total_chunks = max(1, int(np.ceil(len(y) / chunk_samples)))

        recent_ai_probs = []
        det = get_detector()

        for i in range(total_chunks):
            start_idx = i * chunk_samples
            end_idx = min((i + 1) * chunk_samples, len(y))
            chunk_audio = y[start_idx:end_idx]

            if len(chunk_audio) < sr * 0.5:
                continue

            chunk_path = UPLOAD_DIR / f"sim_chunk_{session.id}_{i+1}.wav"
            sf.write(str(chunk_path), chunk_audio, sr)

            try:
                result = await asyncio.to_thread(det.detect, str(chunk_path))
                chunk_ai_prob = result.get("ai_probability", 0.0)

                if result["prediction"] != "ERROR":
                    recent_ai_probs.append(chunk_ai_prob)

                if recent_ai_probs:
                    window = recent_ai_probs[-5:]
                    avg_ai_prob = sum(window) / len(window)
                else:
                    avg_ai_prob = chunk_ai_prob

                if avg_ai_prob >= 65:
                    avg_prediction = "AI_GENERATED"
                elif avg_ai_prob <= 35:
                    avg_prediction = "REAL"
                else:
                    avg_prediction = "UNCERTAIN"

                risk_score = avg_ai_prob
                if risk_score > highest_risk:
                    highest_risk = risk_score

                await websocket.send_json({
                    "prediction": avg_prediction,
                    "confidence": round(avg_ai_prob, 1),
                    "risk_score": round(risk_score, 1),
                    "session_id": session.id,
                    "chunk": i + 1,
                    "total_chunks": total_chunks,
                    "chunk_ai": round(chunk_ai_prob, 1),
                    "details": result.get("explanation", [])
                })
                await asyncio.sleep(0.3)
            except WebSocketDisconnect:
                raise
            except Exception as e:
                print(f"Error analyzing sim chunk {i+1}: {e}")
            finally:
                if chunk_path.exists():
                    try:
                        chunk_path.unlink()
                    except Exception:
                        pass

        # Final summary
        try:
            final_pred = "AI_GENERATED" if highest_risk >= 65 else "REAL" if highest_risk <= 35 else "UNCERTAIN"
            await websocket.send_json({
                "status": "complete",
                "prediction": final_pred,
                "risk_score": round(highest_risk, 1),
                "confidence": round(highest_risk, 1),
                "session_id": session.id,
                "total_chunks": total_chunks,
                "message": "Analysis complete."
            })
        except Exception:
            pass

    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception as e:
        print(f"Simulate call error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        session.ended_at = datetime.utcnow()
        session.risk_score = highest_risk
        db.commit()
        if upload_path.exists():
            try:
                upload_path.unlink()
            except Exception:
                pass


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
