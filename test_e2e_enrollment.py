import os
import io
import wave
import uuid
import json
import pytest
from fastapi.testclient import TestClient

# Mock env vars BEFORE importing main
os.environ["WEB3_PROVIDER_URI"] = ""
os.environ["PRIVATE_KEY"] = ""
os.environ["CONTRACT_ADDRESS"] = ""

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.main import app
from backend.database import Base, engine, SessionLocal
from backend.db_models import User, VoiceEnrollment
from backend.auth import get_password_hash
from backend.blockchain_service import generate_canonical_evidence

client = TestClient(app)

# Helper to create a valid 4-second 16kHz WAV file in memory
def create_valid_wav():
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # 4 seconds of silence/dummy data
        wf.writeframes(b'\x00' * (16000 * 2 * 4))
    buf.seek(0)
    return buf

def create_invalid_wav():
    # 1 second of dummy data (too short)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b'\x00' * (16000 * 2 * 1))
    buf.seek(0)
    return buf

class TestVoiceEnrollmentE2E:
    
    @classmethod
    def setup_class(cls):
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        
        # Create a test user directly in DB
        db = SessionLocal()
        cls.test_user_id = str(uuid.uuid4())
        user = User(id=cls.test_user_id, username="testuser", password_hash=get_password_hash("password123"))
        db.add(user)
        db.commit()
        db.close()
        
        # Get token
        response = client.post("/api/login", data={"username": "testuser", "password": "password123"})
        cls.token = response.json()["access_token"]
        cls.auth_headers = {"Authorization": f"Bearer {cls.token}"}
        cls.enrollment_id = None

    def test_01_authentication_flow(self):
        # Missing auth
        response = client.post("/api/enroll_voice", files={"file": ("test.wav", b"dummy", "audio/wav")})
        assert response.status_code == 401
        
        # Invalid auth
        response = client.post("/api/enroll_voice", files={"file": ("test.wav", b"dummy", "audio/wav")}, headers={"Authorization": "Bearer INVALID"})
        assert response.status_code == 401

    def test_02_audio_too_short(self):
        wav = create_invalid_wav()
        response = client.post("/api/enroll_voice", files={"file": ("short.wav", wav, "audio/wav")}, headers=self.auth_headers)
        assert response.status_code == 400
        assert "Audio too short" in response.json()["detail"]

    def test_03_valid_enrollment(self):
        wav = create_valid_wav()
        response = client.post("/api/enroll_voice", files={"file": ("valid.wav", wav, "audio/wav")}, headers=self.auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "enrollment_id" in data
        assert "evidence_hash" in data
        assert data["status"] == "BLOCKCHAIN_NOT_CONFIGURED" # Since we didn't set ENV
        
        # Save for later tests
        self.__class__.enrollment_id = data["enrollment_id"]

    def test_04_enrollment_retrieval(self):
        response = client.get("/api/enrollment/status", headers=self.auth_headers)
        assert response.status_code == 200
        assert response.json()["enrollment_id"] == self.enrollment_id

    def test_05_re_enrollment_creates_new_version(self):
        wav = create_valid_wav()
        response = client.post("/api/enroll_voice", files={"file": ("valid2.wav", wav, "audio/wav")}, headers=self.auth_headers)
        assert response.status_code == 200
        new_id = response.json()["enrollment_id"]
        assert new_id != self.enrollment_id # Must be a new unique ID
        
        # Check DB versions
        db = SessionLocal()
        enrolls = db.query(VoiceEnrollment).filter(VoiceEnrollment.user_id == self.test_user_id).order_by(VoiceEnrollment.enrollment_version.asc()).all()
        assert len(enrolls) == 2
        assert enrolls[0].enrollment_version == 1
        assert enrolls[1].enrollment_version == 2
        db.close()

    def test_06_sha256_verification(self):
        response = client.get("/api/enrollment/verify", headers=self.auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["verified"] is True
        
        # Tamper with the DB record
        db = SessionLocal()
        latest = db.query(VoiceEnrollment).filter(VoiceEnrollment.user_id == self.test_user_id).order_by(VoiceEnrollment.created_at.desc()).first()
        original_hash = latest.evidence_hash
        latest.evidence_hash = "fake_tampered_hash"
        db.commit()
        db.close()
        
        # Verify should now fail
        response_fail = client.get("/api/enrollment/verify", headers=self.auth_headers)
        assert response_fail.status_code == 200
        assert response_fail.json()["verified"] is False
        assert "TAMPERED" in response_fail.json()["reason"]
        
        # Restore for other tests
        db = SessionLocal()
        latest = db.query(VoiceEnrollment).filter(VoiceEnrollment.user_id == self.test_user_id).order_by(VoiceEnrollment.created_at.desc()).first()
        latest.evidence_hash = original_hash
        db.commit()
        db.close()

    def test_07_user_isolation(self):
        # Create user B
        db = SessionLocal()
        user_b_id = str(uuid.uuid4())
        user_b = User(id=user_b_id, username="user_b", password_hash=get_password_hash("pass"))
        db.add(user_b)
        db.commit()
        db.close()
        
        # Login user B
        res = client.post("/api/login", data={"username": "user_b", "password": "pass"})
        token_b = res.json()["access_token"]
        
        # Try to get User A's enrollment (Should return 404 since it filters by current_user_id)
        status_res = client.get("/api/enrollment/status", headers={"Authorization": f"Bearer {token_b}"})
        assert status_res.status_code == 404

if __name__ == "__main__":
    print("Running E2E tests for Voice Enrollment...")
    pytest.main(["-v", __file__])
