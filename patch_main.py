import re
import sys

def modify_main():
    with open('backend/main.py', 'r') as f:
        content = f.read()

    imports_to_add = """
from fastapi import Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.db_models import User, VoiceEnrollment
from backend.auth import get_password_hash, verify_password, create_access_token, get_current_user_id
from backend.blockchain_service import generate_canonical_evidence, anchor_hash_to_blockchain, is_blockchain_configured
import uuid
import librosa
"""
    # Add imports after the first batch of FastAPI imports
    content = content.replace("from fastapi.middleware.cors import CORSMiddleware", "from fastapi.middleware.cors import CORSMiddleware\n" + imports_to_add)

    auth_routes = """
# ─── Auth Endpoints ───────────────────────────────────────────────────

@app.post("/api/register")
async def register(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(form_data.password)
    user_id = str(uuid.uuid4())
    new_user = User(id=user_id, username=form_data.username, password_hash=hashed_password)
    db.add(new_user)
    db.commit()
    return {"message": "User created successfully", "user_id": user_id}

@app.post("/api/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": user.id})
    return {"access_token": access_token, "token_type": "bearer"}

"""
    # Insert auth routes before "Blockchain Voice ID Endpoints"
    content = content.replace("# ─── Blockchain Voice ID Endpoints", auth_routes + "\n# ─── Blockchain Voice ID Endpoints")

    enroll_voice_new = """
@app.post("/api/enroll_voice")
async def enroll_voice_endpoint(
    file: UploadFile = File(...),
    current_user_id: str = Depends(get_current_user_id),
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
            status="BLOCKCHAIN_PENDING"
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
    current_user_id: str = Depends(get_current_user_id),
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
    current_user_id: str = Depends(get_current_user_id),
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

"""

    # Replace old enroll_voice block using regex
    # The old block goes from `@app.post("/api/enroll_voice")` down to `return {...}` 
    pattern = re.compile(r'@app\.post\("/api/enroll_voice"\).*?return \{.*?\}', re.DOTALL)
    content = pattern.sub(enroll_voice_new.strip(), content)

    with open('backend/main.py', 'w') as f:
        f.write(content)
        
    print("Successfully patched main.py")

if __name__ == "__main__":
    modify_main()
