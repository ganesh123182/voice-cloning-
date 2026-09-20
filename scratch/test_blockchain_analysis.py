import sqlite3
import os
import json
import hashlib
from backend.blockchain import verify_integrity, hash_voiceprint, extract_voiceprint, verify_speaker, LEDGER_FILE
from backend.blockchain_service import generate_canonical_evidence, is_blockchain_configured

print("=" * 60)
print("1. LEDGER INTEGRITY AUDIT")
print("=" * 60)
is_valid, msg = verify_integrity()
print(f"Overall verify_integrity(): {is_valid} -> {msg}")

with open(LEDGER_FILE, 'r') as f:
    ledger = json.load(f)

print(f"Total Enrolled Users in Ledger: {len(ledger)}")
all_match = True
for uid, entry in ledger.items():
    vp = entry.get("voiceprint")
    stored_hash = entry.get("sha256_hash") or entry.get("voice_hash")
    recalc_hash = hash_voiceprint(vp)
    dim = len(vp) if isinstance(vp, list) else 0
    if stored_hash != recalc_hash:
        print(f"  [MISMATCH] User {uid}: stored={stored_hash[:10]}... != recalc={recalc_hash[:10]}...")
        all_match = False
    else:
        print(f"  [OK] User: {uid:15} | Vector Dim: {dim:3} | SHA-256: {stored_hash[:16]}... Match: True")

print(f"Ledger Cryptographic Match: {'PERFECT (100%)' if all_match else 'FAILED'}")

print("\n" + "=" * 60)
print("2. TAMPER RESISTANCE TEST")
print("=" * 60)
# Simulate tampering with a single float in a copy of voiceprint
test_vp = list(ledger["demo_user"]["voiceprint"])
original_hash = hash_voiceprint(test_vp)
test_vp[0] += 0.000001
tampered_hash = hash_voiceprint(test_vp)
print(f"Original Voiceprint Hash : {original_hash}")
print(f"Tampered (+0.000001) Hash: {tampered_hash}")
print(f"Tamper Detected via Avalanche Effect: {original_hash != tampered_hash}")

print("\n" + "=" * 60)
print("3. SQLITE DATABASE ENROLLMENTS AUDIT")
print("=" * 60)
db_path = "backend/voice_enrollment.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT enrollment_id, user_id, enrollment_version, model_version, evidence_hash, voice_hash, status, created_at FROM voice_enrollments")
    rows = cur.fetchall()
    print(f"Total Database Enrollments: {len(rows)}")
    for r in rows:
        eid, uid, e_ver, m_ver, ev_hash, v_hash, status, created = r
        # Verify canonical evidence hash recreation
        expected_ev_hash = generate_canonical_evidence(
            enrollment_id=eid,
            user_id=uid,
            model_version=m_ver,
            enrollment_version=e_ver,
            timestamp=created
        )
        match = (expected_ev_hash == ev_hash)
        print(f"  Enrollment: {eid[:8]}... | User: {uid:15} | Status: {status:20} | Evidence Hash Match: {match}")
    conn.close()
else:
    print("Database backend/voice_enrollment.db not found.")

print("\n" + "=" * 60)
print("4. ON-CHAIN WEB3 READINESS CHECK")
print("=" * 60)
configured = is_blockchain_configured()
print(f"Web3 RPC & Smart Contract Configured: {configured}")
if not configured:
    print("  -> System operating in cryptographic Local Ledger mode (Tamper-evident SHA-256).")
    print("  -> Ready to connect to Ethereum / Polygon / Ganache EVM RPC when WEB3_PROVIDER_URI & CONTRACT_ADDRESS are set.")

print("\n" + "=" * 60)
print("5. REAL-TIME VERIFICATION RUNTIME CHECK")
print("=" * 60)
import numpy as np
dummy_audio = np.random.randn(16000).astype(np.float32)
try:
    is_ver, sim = verify_speaker("demo_user", dummy_audio, sr=16000)
    print(f"Verification test against 'demo_user' with random noise:")
    print(f"  is_verified: {is_ver} (Expected: False), similarity: {sim:.4f}")
except Exception as e:
    print(f"  Error in verify_speaker: {e}")

print("=" * 60)
