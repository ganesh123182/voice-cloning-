import os
import json
import hashlib
import numpy as np
import librosa
from scipy.spatial.distance import cosine
from typing import Dict, Union, Tuple
import torch
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

try:
    from speechbrain.inference.speaker import EncoderClassifier
    # Pre-load ECAPA model globally to save time during verification
    # Using local dir to avoid symlink crash on Windows
    ecapa_classifier = EncoderClassifier.from_hparams(
        source="backend/models/spkrec-ecapa-voxceleb", 
        savedir="backend/models/spkrec-ecapa-voxceleb"
    )
except Exception as e:
    print(f"[WARN] Failed to load ECAPA-TDNN model: {e}")
    ecapa_classifier = None

LEDGER_FILE = os.path.join(os.path.dirname(__file__), "ledger.json")

def extract_voiceprint(audio_input: Union[str, np.ndarray], sr: int = 16000) -> list:
    """
    Extracts a secure Voice ID (voiceprint) from audio using ECAPA-TDNN (192-dim vector).
    """
    if ecapa_classifier is None:
        raise RuntimeError("ECAPA-TDNN model is not loaded. Cannot extract voiceprint.")

    if isinstance(audio_input, str):
        import librosa
        import torch
        # librosa automatically handles resampling and mono conversion
        # bypassing the torchaudio/ffmpeg DLL missing crash on Windows
        y, fs = librosa.load(audio_input, sr=sr, mono=True)
        signal = torch.tensor(y).unsqueeze(0)
    else:
        # np.ndarray
        signal = torch.from_numpy(audio_input).unsqueeze(0)
        
    with torch.no_grad():
        embeddings = ecapa_classifier.encode_batch(signal)
    
    # Return 192-d float list
    return embeddings.squeeze().tolist()

def hash_voiceprint(voiceprint: list) -> str:
    """
    Generates a SHA-256 hash of the voiceprint to act as a blockchain-style signature.
    """
    # Serialize with strict formatting for consistent hashing
    serialized = json.dumps(voiceprint, separators=(',', ':'))
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

def enroll_user(user_id: str, audio_path: str) -> Dict:
    """
    Enrolls a user by extracting their voiceprint, hashing it, and saving to the ledger.
    """
    voiceprint = extract_voiceprint(audio_path)
    secure_hash = hash_voiceprint(voiceprint)
    
    entry = {
        "user_id": user_id,
        "voiceprint": voiceprint,
        "sha256_hash": secure_hash
    }
    
    # Load existing ledger
    ledger = {}
    if os.path.exists(LEDGER_FILE):
        try:
            with open(LEDGER_FILE, 'r') as f:
                ledger = json.load(f)
        except Exception:
            pass
            
    # Add new entry and save
    ledger[user_id] = entry
    with open(LEDGER_FILE, 'w') as f:
        json.dump(ledger, f, indent=4)
        
    return entry

def verify_integrity() -> Tuple[bool, str]:
    """
    Audits the local ledger. Re-calculates the SHA-256 hash of every stored 
    voiceprint and compares it against the stored hash. 
    Returns (True, "Valid") or (False, "Tampered user: {user_id}").
    """
    if not os.path.exists(LEDGER_FILE):
        return True, "No ledger found."
        
    try:
        with open(LEDGER_FILE, 'r') as f:
            ledger = json.load(f)
    except Exception as e:
        return False, f"Ledger corrupted: {str(e)}"
        
    for user_id, data in ledger.items():
        stored_hash = data.get("sha256_hash")
        stored_voiceprint = data.get("voiceprint")
        
        # Re-calculate hash
        actual_hash = hash_voiceprint(stored_voiceprint)
        
        if stored_hash != actual_hash:
            return False, f"🚨 TAMPERING DETECTED! User '{user_id}' voiceprint hash mismatch."
            
    return True, "Blockchain ledger integrity verified."

def verify_speaker(user_id: str, live_audio: np.ndarray, sr: int = 16000) -> Tuple[Union[bool, None], float]:
    """
    Compares live audio chunk against the enrolled user's voiceprint using Cosine Similarity.
    Returns (is_verified, similarity_score). If user is not enrolled, returns (None, 0.0).
    """
    if not os.path.exists(LEDGER_FILE):
        return None, 0.0
        
    with open(LEDGER_FILE, 'r') as f:
        ledger = json.load(f)
        
    if user_id not in ledger:
        return None, 0.0 # User not enrolled
        
    enrolled_vp = np.array(ledger[user_id]["voiceprint"])
    live_vp = np.array(extract_voiceprint(live_audio, sr))
    
    # Compute Cosine Similarity (1 - cosine distance)
    similarity = 1 - cosine(enrolled_vp, live_vp)
    
    # Calibrated threshold for ECAPA-TDNN (VoxCeleb)
    # Typical threshold for this model is ~0.25 - 0.30
    is_verified = similarity > 0.35
    
    return is_verified, similarity
