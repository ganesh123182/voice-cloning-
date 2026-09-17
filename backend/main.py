"""
FastAPI Backend - AI Voice Cloning + Synthetic Voice Detection
Main application entry point.
"""
import os
import uuid
import time
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from fastapi import Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.db_models import User, VoiceEnrollment
from backend.auth import get_password_hash, verify_password, create_access_token, get_current_user_id
from backend.blockchain_service import generate_canonical_evidence, anchor_hash_to_blockchain, is_blockchain_configured
import uuid
import librosa


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
    file: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None),
):
    """
    Analyze an audio file to determine if it's AI-generated or human.
    Returns probabilistic results with confidence scores.
    """
    upload = file or audio_file
    if upload is None:
        raise HTTPException(status_code=400, detail="No audio file provided. Please upload an audio file.")

    # Save uploaded audio
    try:
        filepath = await save_upload(upload, prefix="detect")
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

import uuid
from fastapi import Depends, HTTPException, BackgroundTasks, Form, Query, Request
from fastapi.security import OAuth2PasswordRequestForm
from backend.auth import get_password_hash, verify_password, create_access_token, get_current_user_id
from backend.db_models import User, VoiceEnrollment

@app.post("/api/register")
async def register(
    request: Request,
    username: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    full_name: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    # Support both Form-encoded and JSON body
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            username = body.get("username") or body.get("email") or body.get("phone") or username
            password = body.get("password") or password
            email = body.get("email", email)
            phone = body.get("phone", phone)
            full_name = body.get("full_name", full_name)
        except Exception:
            pass

    user_identifier = (username or email or phone or "").strip()
    if not user_identifier or not password:
        raise HTTPException(status_code=400, detail="Username/Email and password are required")

    # Check if user already exists
    existing = db.query(User).filter(
        (User.username == user_identifier) | 
        (User.email == user_identifier) | 
        (User.phone == user_identifier)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Account already exists for this email or username")

    new_user = User(
        id=str(uuid.uuid4()),
        username=user_identifier,
        password_hash=get_password_hash(password),
        full_name=full_name or user_identifier.split("@")[0],
        email=email or (user_identifier if "@" in user_identifier else None),
        phone=phone or (user_identifier if not "@" in user_identifier else None)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    access_token = create_access_token(data={"sub": new_user.id})
    return {
        "message": "User registered successfully",
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": new_user.id,
        "username": new_user.username,
        "full_name": new_user.full_name
    }

@app.post("/api/login")
async def login(
    request: Request,
    username: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            username = body.get("username") or body.get("email") or body.get("phone") or username
            password = body.get("password") or password
        except Exception:
            pass

    user_identifier = (username or "").strip()
    if not user_identifier or not password:
        raise HTTPException(status_code=400, detail="Username/Email and password are required")

    user = db.query(User).filter(
        (User.username == user_identifier) |
        (User.email == user_identifier) |
        (User.phone == user_identifier)
    ).first()

    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect email, username, or password")

    access_token = create_access_token(data={"sub": user.id})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username,
        "full_name": user.full_name or user.username
    }

async def resolve_user_id_post(
    request: Request,
    user_id: str = Form(None),
    db: Session = Depends(get_db)
):
    """For POST endpoints that receive user_id via multipart form data."""
    if user_id: return user_id
    # Fallback to JWT token
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            return get_current_user_id(token)
        except Exception:
            pass
    raise HTTPException(status_code=401, detail="Missing user_id or valid token")

async def resolve_user_id_get(
    request: Request,
    user_id: str = Query(None),
    db: Session = Depends(get_db)
):
    """For GET endpoints that receive user_id via query parameter."""
    if user_id: return user_id
    # Fallback to JWT token
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            return get_current_user_id(token)
        except Exception:
            pass
    raise HTTPException(status_code=401, detail="Missing user_id query param or valid token")

# ─── Blockchain Voice ID Endpoints ────────────────────────────────────

from backend.blockchain import enroll_user, verify_integrity, verify_speaker

@app.post("/api/enroll_voice")
async def enroll_voice_endpoint(
    file: UploadFile = File(...),
    current_user_id: str = Depends(resolve_user_id_post),
    db: Session = Depends(get_db)
):
    try:
        # Validate Audio
        filepath = await save_upload(file, prefix="enroll")
        y, sr = librosa.load(filepath, sr=16000, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < 3.0:
            raise HTTPException(status_code=400, detail="Audio too short. Minimum 3 seconds required.")
        
        # Check idempotency / retries based on current user active enrollment? 
        # Actually user wants versions.
        
        # Extract ECAPA-TDNN Embedding (used for WHO is speaking, not Wav2Vec2)
        from backend.blockchain import extract_voiceprint
        voiceprint = extract_voiceprint(filepath)
        
        # Instead of storing raw voiceprint on blockchain, we store in DB off-chain.
        enrollment_id = str(uuid.uuid4())
        
        # Calculate Enrollment Version
        existing_enrollments = db.query(VoiceEnrollment).filter(VoiceEnrollment.user_id == current_user_id).count()
        enroll_version = existing_enrollments + 1
        
        # Canonical Evidence
        timestamp = datetime.utcnow().isoformat()
        evidence_hash = generate_canonical_evidence(
            enrollment_id=enrollment_id,
            user_id=current_user_id,
            model_version="ecapa-tdnn-voxceleb",
            enrollment_version=enroll_version,
            timestamp=timestamp
        )
        
        # Save to DB
        enrollment = VoiceEnrollment(
            enrollment_id=enrollment_id,
            user_id=current_user_id,
            enrollment_version=enroll_version,
            evidence_hash=evidence_hash,
            status="BLOCKCHAIN_PENDING",
            created_at=datetime.fromisoformat(timestamp)
        )
        db.add(enrollment)
        db.commit()
        
        # Blockchain Anchor
        tx_hash = None
        status = "BLOCKCHAIN_NOT_CONFIGURED"
        if is_blockchain_configured():
            try:
                tx_hash = anchor_hash_to_blockchain(evidence_hash, enrollment_id)
                status = "BLOCKCHAIN_CONFIRMED"
                enrollment.blockchain_tx_hash = tx_hash
                enrollment.blockchain_network = "EVM"
            except Exception as e:
                print(f"[ERROR] Blockchain anchoring failed: {e}")
                status = "BLOCKCHAIN_FAILED"
        
        enrollment.status = status
        db.commit()
        
        log_activity("voice_enrollment", f"User: {current_user_id} | Hash: {evidence_hash[:8]}... | Status: {status}", "success")
        
        # Note: In a real app we'd save the voiceprint to a secure vector DB. 
        # For now, we still save it to the legacy ledger just so `verify_speaker` works without breaking existing code.
        from backend.blockchain import hash_voiceprint, LEDGER_FILE
        import json, os
        ledger = {}
        if os.path.exists(LEDGER_FILE):
            try:
                with open(LEDGER_FILE, 'r') as f: ledger = json.load(f)
            except: pass
        ledger[current_user_id] = {
            "user_id": current_user_id,
            "voiceprint": voiceprint,
            "sha256_hash": evidence_hash
        }
        with open(LEDGER_FILE, 'w') as f: json.dump(ledger, f, indent=4)
        
        return {
            "success": True,
            "message": "Voiceprint enrolled successfully.",
            "enrollment_id": enrollment_id,
            "status": status,
            "evidence_hash": evidence_hash,
            "hash": evidence_hash,
            "blockchain_tx_hash": tx_hash
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error during enrollment.")

@app.get("/api/enrollment/status")
async def get_enrollment_status(
    current_user_id: str = Depends(resolve_user_id_get),
    db: Session = Depends(get_db)
):
    enrollment = db.query(VoiceEnrollment).filter(VoiceEnrollment.user_id == current_user_id).order_by(VoiceEnrollment.created_at.desc()).first()
    if not enrollment:
        raise HTTPException(status_code=404, detail="No enrollment found")
        
    return {
        "enrollment_id": enrollment.enrollment_id,
        "status": enrollment.status,
        "evidence_hash": enrollment.evidence_hash,
        "blockchain_tx_hash": enrollment.blockchain_tx_hash,
        "created_at": enrollment.created_at.isoformat()
    }

@app.get("/api/enrollment/verify")
async def verify_enrollment(
    current_user_id: str = Depends(resolve_user_id_get),
    db: Session = Depends(get_db)
):
    enrollment = db.query(VoiceEnrollment).filter(VoiceEnrollment.user_id == current_user_id).order_by(VoiceEnrollment.created_at.desc()).first()
    if not enrollment:
        raise HTTPException(status_code=404, detail="No enrollment found")
        
    # Recompute hash
    expected_hash = generate_canonical_evidence(
        enrollment_id=enrollment.enrollment_id,
        user_id=enrollment.user_id,
        model_version=enrollment.model_version,
        enrollment_version=enrollment.enrollment_version,
        timestamp=enrollment.created_at.isoformat()
    )
    
    if expected_hash != enrollment.evidence_hash:
        return {"verified": False, "reason": "Database evidence hash mismatch (TAMPERED)"}
        
    if is_blockchain_configured():
        from backend.blockchain_service import verify_hash_on_blockchain
        try:
            on_chain_hash = verify_hash_on_blockchain(enrollment.enrollment_id)
            if on_chain_hash != enrollment.evidence_hash:
                return {"verified": False, "reason": "Blockchain hash mismatch"}
        except Exception as e:
            return {"verified": False, "reason": f"Blockchain verification failed: {e}"}
            
    return {"verified": True, "evidence_hash": expected_hash, "blockchain_status": enrollment.status}

@app.get("/api/verify_integrity")
async def verify_integrity_endpoint():
    """
    Audits the blockchain ledger to detect any tampered voiceprints.
    """
    is_valid, message = verify_integrity()
    return {
        "success": is_valid,
        "integrity_status": "Valid" if is_valid else "Tampered",
        "message": message
    }

# ─── Live Monitoring WebSocket ────────────────────────────────────────

@app.websocket("/api/monitoring/live")
async def live_monitoring(websocket: WebSocket, token: str, caller_number: str = "Unknown", caller_name: str = "Unknown"):
    """
    WebSocket endpoint for real-time live call monitoring.
    Receives audio chunks from the Android app and returns deepfake probabilities.
    """
    await websocket.accept()
    
    # In a real app, validate the token here.
    if not token:
        await websocket.close(code=1008)
        return

    import numpy as np
    import torch
    import collections
    
    # Store last 3 risk scores for temporal smoothing
    recent_scores = collections.deque(maxlen=3)
    
    try:
        while True:
            # Receive binary audio chunk (16-bit PCM from Android AudioRecord)
            data = await websocket.receive_bytes()
            if not data:
                continue
                
            import wave, tempfile, os, librosa
            import scipy.signal as sig
            
            # ── Convert raw PCM to float32 ──
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            # ── Silence guard BEFORE AGC ──
            raw_peak = np.max(np.abs(audio_data))
            if raw_peak < 0.015:
                print(f"[LIVE] Silence/noise ignored! raw_peak={raw_peak:.4f}")
                risk_score = 0.0
                label = "Silence"
                is_alert = False
                suggestion = "Waiting for caller..."
            else:
                # ── AGC: Normalize volume ──
                audio_data = audio_data * min(0.6 / raw_peak, 30.0)  # Cap at 30x gain
                sr = 16000

                # ── Software Bandpass Filter: 300Hz-3400Hz (phone speech band) ──
                b, a = sig.butter(4, [300/(sr/2), 3400/(sr/2)], btype='bandpass')
                audio_clean = sig.filtfilt(b, a, audio_data).astype(np.float32)

                # ── Mild Noise Gate: 0.005 (removes room hum, keeps speech) ──
                audio_clean = np.where(np.abs(audio_clean) < 0.005, 0.0, audio_clean).astype(np.float32)

                reasons = []

                # NOTE: LIVE MIC RECORDING AUTO-SAVE DISABLED TO PREVENT DATA LEAKAGE

                
                # Check VAD (Speech Ratio) before saving to avoid saving pure silence
                frame_len = int(0.030 * sr)  # 30ms
                n_frames = len(audio_clean) // frame_len
                speech_frames = sum(
                    1 for i in range(n_frames)
                    if float(np.sqrt(np.mean(audio_clean[i*frame_len:(i+1)*frame_len]**2))) > 0.005
                )
                speech_ratio = speech_frames / max(n_frames, 1)
                
                # Expose vad_val for Layer 8 and Layer 9 checks below
                vad_val = speech_ratio

                if speech_ratio > 0.20:
                    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                    try:
                        with os.fdopen(tmp_fd, 'wb') as f:
                            with wave.open(f, 'wb') as wf:
                                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
                                # Pass RAW audio_data to the ML model (matches training distribution)
                                out_data = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                                wf.writeframes(out_data)
                        
                        # ── LAYER: DEEP LEARNING (Wav2Vec2) ──
                        # Inference runs in a thread pool to avoid blocking the WS event loop!
                        ml_result = await asyncio.to_thread(detector.analyze, tmp_path)
                        ai_prob = float(ml_result.get("ai_probability", 0.0))
                        
                        if ml_result.get("prediction") == "error" or ml_result.get("prediction") == "insufficient_data":
                            # Ignore short frames or errors
                            risk_score = 50.0
                        else:
                            raw_ml_risk = ai_prob
                            recent_scores.append(raw_ml_risk)
                            risk_score = sum(recent_scores) / len(recent_scores) # Temporal smoothing

                        if risk_score > 50:
                            reasons.append(f"⚠ Wav2Vec2 Model Flagged as Synthetic ({risk_score:.1f}%)")
                        else:
                            if len(recent_scores) > 0:
                                reasons.append(f"✅ Wav2Vec2 Model classified as Human ({(100-risk_score):.1f}%)")
                                
                        # --- ADDED FOR DATA COLLECTION ---
                        import shutil
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        dest_path = f"E:/VoiceDeepfakeAI/collected_audio/unlabeled/live_chunk_{timestamp}.wav"
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        try:
                            shutil.copy2(tmp_path, dest_path)
                        except Exception as e:
                            print(f"[DEBUG] Failed to save chunk: {e}")
                        # ---------------------------------
                            
                    except Exception as e:
                        print(f"[ERROR] Live processing failed: {e}")
                        risk_score = 50.0
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                else:
                    risk_score = 0.0
                    is_alert = False
                    reasons.append("Low speech ratio - treated as silence.")

                log_line = f"[LIVE] VAD={speech_ratio:.2f} => RISK={risk_score:.1f}"
                print(log_line)
                is_alert = risk_score > 50.0


                # ── LAYER 8: NLP Scam Keyword Detection ──
                transcript = ""
                try:
                    import speech_recognition as sr
                    
                    def do_stt(raw_bytes):
                        r = sr.Recognizer()
                        # Our chunks are 16kHz, 16-bit PCM, Mono
                        audio_obj = sr.AudioData(raw_bytes, 16000, 2)
                        return r.recognize_google(audio_obj)
                    
                    # Only transcribe if VAD indicates speech to save API calls
                    if vad_val > 0.15:
                        transcript = await asyncio.to_thread(do_stt, data)
                        print(f"[LIVE] Transcript: '{transcript}'")
                        
                        scam_keywords = ["otp", "bank account", "police", "urgent", "transfer", "password", "pin", "verification", "send money", "account details"]
                        transcript_lower = transcript.lower()
                        found_keywords = [kw for kw in scam_keywords if kw in transcript_lower]
                        
                        # If a keyword is found and the acoustic score is at least slightly suspicious
                        if found_keywords and risk_score > 40.0:
                            risk_score = 100.0
                            is_alert = True
                            reasons.append(f"🧨 SCAM CONTEXT: Detected words ({', '.join(found_keywords)})")
                except Exception as e:
                    pass

                # ── LAYER 9: Speaker Verification (Identity Check) ──
                # Check if the person speaking matches the enrolled user for this token
                speaker_verified = False
                try:
                    from backend.blockchain import verify_speaker
                    # Using token as user_id for demonstration purposes
                    # If it's a silence chunk, we don't verify
                    if risk_score > 0.0 and vad_val > 0.10:
                        is_verified, sim_score = verify_speaker(token, audio_clean, sr=16000)
                        
                        if is_verified is True:
                            speaker_verified = True
                        elif is_verified is False:
                            # They ARE enrolled, but the voice didn't match!
                            reasons.append(f"🔒 Identity Mismatch: Unauthorized Speaker (Sim: {sim_score:.2f})")
                            # Boost risk score slightly because it's an unrecognized speaker
                            if risk_score < 100.0:
                                risk_score = min(risk_score + 15, 100.0)
                        else:
                            # is_verified is None (User not enrolled)
                            pass 
                except Exception:
                    pass

                if risk_score >= 100.0:
                    label = "SCAM CALL DETECTED"
                    suggestion = "CRITICAL WARNING: Voice clone attempting social engineering! Hang up immediately."
                elif risk_score > 56:
                    label = "FAKE VOICE DETECTED"
                    suggestion = "WARNING: AI-generated voice! Do NOT share OTP or transfer money!"
                elif risk_score > 45:
                    label = "Suspicious Audio"
                    suggestion = "Caution: Unusual voice patterns detected"
                else:
                    label = "Human Voice"
                    suggestion = "Safe to proceed"
            
            
            # Map speaker field to values the Android FloatingHUD expects:
            #   "user"    → VERIFIED USER (green)
            #   "caller"  → analyzed caller voice (color depends on risk_score)
            #   "silence" → NO SPEECH DETECTED (gray)
            if risk_score == 0.0:
                speaker_label = "silence"
            elif speaker_verified:
                speaker_label = "user"
            else:
                speaker_label = "caller"

            # Compute speaker_match percentage for Android UI
            speaker_match_pct = 0
            try:
                if risk_score > 0.0 and vad_val > 0.10:
                    from backend.blockchain import verify_speaker as vs_check
                    _, sim = vs_check(token, audio_data, sr=16000)
                    speaker_match_pct = max(0, min(100, int(sim * 100)))
            except Exception:
                pass

            await websocket.send_json({
                "speaker": speaker_label,
                "risk_score": round(float(risk_score), 1),
                "label": label,
                "is_alert": bool(is_alert),
                "suggestion": suggestion,
                "explainability_reasons": reasons if risk_score > 0.0 else [],
                "transcript": transcript,
                "speaker_match": speaker_match_pct,
                "caller_name": caller_name,
                "caller_number": caller_number
            })
            
    except WebSocketDisconnect:
        print("Live monitoring client disconnected.")
    except Exception as e:
        print(f"Error in live monitoring: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


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
    # Since this is a prototype, return some hardcoded demo data that perfectly matches the UI,
    # and any real activity logs if they exist.
    
    mock_activity = [
        {
            "id": "mock_1",
            "caller_name": "Call analyzed \u2014 Safe",
            "phone_number": "+91 98765 43210",
            "status": "Safe",
            "timestamp": "10:24 AM"
        },
        {
            "id": "mock_2",
            "caller_name": "Suspicious voice detected",
            "phone_number": "+91 91234 56789",
            "status": "Blocked",
            "timestamp": "Yesterday"
        },
        {
            "id": "mock_3",
            "caller_name": "Verification completed",
            "phone_number": "+91 87654 32109",
            "status": "Verified",
            "timestamp": "12 Sep"
        }
    ]
    
    return {
        "threats_blocked": 12,
        "recent_activity": mock_activity
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
        reload=False,
        log_level="info",
    )
