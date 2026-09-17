import os
import pytest
import io
import wave
import uuid
from fastapi.testclient import TestClient

# Must set ENV before importing main
os.environ["WEB3_PROVIDER_URI"] = "http://127.0.0.1:8545"
# Hardhat Account #0 private key
os.environ["PRIVATE_KEY"] = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
# Contract deployed
os.environ["CONTRACT_ADDRESS"] = "0x5FbDB2315678afecb367f032d93F642f64180aa3"
os.environ["CHAIN_ID"] = "31337"

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.main import app
from backend.database import Base, engine, SessionLocal
from backend.db_models import User, VoiceEnrollment
from backend.auth import get_password_hash

client = TestClient(app)

def create_valid_wav():
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b'\x00' * (16000 * 2 * 4))
    buf.seek(0)
    return buf

class TestRealBlockchain:
    @classmethod
    def setup_class(cls):
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        db = SessionLocal()
        cls.test_user_id = str(uuid.uuid4())
        user = User(id=cls.test_user_id, username="bc_testuser", password_hash=get_password_hash("password123"))
        db.add(user)
        db.commit()
        db.close()
        
        response = client.post("/api/login", data={"username": "bc_testuser", "password": "password123"})
        cls.token = response.json()["access_token"]
        cls.auth_headers = {"Authorization": f"Bearer {cls.token}"}

    def test_01_real_blockchain_anchoring(self):
        # Create enrollment
        wav = create_valid_wav()
        response = client.post("/api/enroll_voice", files={"file": ("test.wav", wav, "audio/wav")}, headers=self.auth_headers)
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["status"] == "BLOCKCHAIN_CONFIRMED"
        assert "blockchain_tx_hash" in data
        assert data["blockchain_tx_hash"].startswith("0x") # Real tx hash returned
        
        self.__class__.enrollment_id = data["enrollment_id"]
        self.__class__.evidence_hash = data["evidence_hash"]

    def test_02_blockchain_verification(self):
        # Hit the verify endpoint which fetches from contract
        response = client.get("/api/enrollment/verify", headers=self.auth_headers)
        assert response.status_code == 200
        data = response.json()
        
        assert data["verified"] is True
        assert data["blockchain_status"] == "MATCH"
        assert data["evidence_hash"] == self.evidence_hash

    def test_03_tamper_detection(self):
        # Tamper the DB
        db = SessionLocal()
        latest = db.query(VoiceEnrollment).filter(VoiceEnrollment.user_id == self.test_user_id).first()
        original_hash = latest.evidence_hash
        latest.evidence_hash = "fake_tampered_hash"
        db.commit()
        db.close()
        
        response = client.get("/api/enrollment/verify", headers=self.auth_headers)
        assert response.status_code == 200
        data = response.json()
        
        assert data["verified"] is False
        assert "TAMPERED" in data["reason"] or "MISMATCH" in data["reason"]
        
        # Restore
        db = SessionLocal()
        latest = db.query(VoiceEnrollment).filter(VoiceEnrollment.user_id == self.test_user_id).first()
        latest.evidence_hash = original_hash
        db.commit()
        db.close()

if __name__ == "__main__":
    pytest.main(["-v", __file__])
