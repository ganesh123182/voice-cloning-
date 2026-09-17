"""
End-to-End Functional Test for Deepfake Voice Verification Feature
Validates backend /api/detect contract with Android app client requirements.
"""
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.main import app

client = TestClient(app)

def test_detect_synthetic_voice_with_file_param():
    """Test detecting synthetic voice using 'file' multipart param (used by Android Retrofit)."""
    assert os.path.exists("test.wav"), "test.wav must exist"
    
    with open("test.wav", "rb") as f:
        response = client.post(
            "/api/detect",
            files={"file": ("test.wav", f, "audio/wav")}
        )
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    
    print("\n[TEST 1] Synthetic Voice Detection Result:")
    print(data)
    
    assert data["success"] is True
    assert "prediction" in data
    assert "confidence" in data
    assert "ai_probability" in data
    assert "risk_level" in data
    assert "scores" in data
    assert "details" in data
    assert data["ai_probability"] >= 50.0, "test.wav should have high AI probability"
    assert data["prediction"] in ("likely_ai_generated", "AI_GENERATED")
    assert data["risk_level"] in ("CRITICAL", "HIGH", "MEDIUM")


def test_detect_with_audio_file_param():
    """Test backwards compatibility with 'audio_file' multipart param."""
    assert os.path.exists("voice1.wav"), "voice1.wav must exist"
    
    with open("voice1.wav", "rb") as f:
        response = client.post(
            "/api/detect",
            files={"audio_file": ("voice1.wav", f, "audio/wav")}
        )
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    
    print("\n[TEST 2] 'audio_file' param compatibility result:")
    print(data)
    
    assert data["success"] is True
    assert data["ai_probability"] > 0


def test_detect_real_voice_sample():
    """Test detecting authentic human voice sample."""
    sample_path = "backend/models/spkrec-ecapa-voxceleb/example1.wav"
    assert os.path.exists(sample_path), f"{sample_path} must exist"
    
    with open(sample_path, "rb") as f:
        response = client.post(
            "/api/detect",
            files={"file": ("example1.wav", f, "audio/wav")}
        )
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    
    print("\n[TEST 3] Real Voice Detection Result:")
    print(data)
    
    assert data["success"] is True
    assert "prediction" in data
    assert "confidence" in data
    assert "ai_probability" in data
    assert "risk_level" in data


def test_detect_no_file_provided():
    """Test that missing audio file returns clean 400 Bad Request error."""
    response = client.post("/api/detect", data={})
    assert response.status_code == 400
    assert "detail" in response.json()
    print("\n[TEST 4] Missing file properly rejected with 400:", response.json()["detail"])


def test_detect_unsupported_format():
    """Test that non-audio extension returns clean 400 Bad Request error."""
    response = client.post(
        "/api/detect",
        files={"file": ("malicious.exe", b"dummy_content", "application/octet-stream")}
    )
    assert response.status_code == 400
    print("\n[TEST 5] Invalid format properly rejected with 400:", response.json()["detail"])


if __name__ == "__main__":
    print("=== RUNNING DEEPFAKE VOICE VERIFICATION E2E TESTS ===")
    test_detect_synthetic_voice_with_file_param()
    test_detect_with_audio_file_param()
    test_detect_real_voice_sample()
    test_detect_no_file_provided()
    test_detect_unsupported_format()
    print("\n>>> ALL 5 END-TO-END TESTS PASSED SUCCESSFULLY! <<<")
