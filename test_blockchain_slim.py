import os
import uuid

# Must set ENV
os.environ["WEB3_PROVIDER_URI"] = "http://127.0.0.1:8545"
os.environ["PRIVATE_KEY"] = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
os.environ["CONTRACT_ADDRESS"] = "0x5FbDB2315678afecb367f032d93F642f64180aa3"
os.environ["CHAIN_ID"] = "31337"

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.blockchain_service import (
    anchor_hash_to_blockchain, 
    verify_hash_on_blockchain, 
    generate_canonical_evidence
)

def run_slim_test():
    print("--- Starting Slim Blockchain Test ---")
    enrollment_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    
    # Generate canonical hash
    evidence_hash = generate_canonical_evidence(
        enrollment_id=enrollment_id,
        user_id=user_id,
        model_version="1.0",
        enrollment_version=1,
        timestamp="2026-09-16T12:00:00Z"
    )
    print(f"[1] Canonical Hash Generated: {evidence_hash}")
    
    # Anchor to Hardhat network
    tx_hash = anchor_hash_to_blockchain(evidence_hash, enrollment_id)
    print(f"[2] Transaction Submitted to Hardhat. TX Hash: {tx_hash}")
    
    # Read back from contract
    retrieved_hash = verify_hash_on_blockchain(enrollment_id)
    print(f"[3] Hash Retrieved from Contract: {retrieved_hash}")
    
    if evidence_hash == retrieved_hash:
        print("[4] SUCCESS: On-chain hash strictly matches canonical hash!")
    else:
        print("[4] FAIL: Hashes do not match!")

if __name__ == "__main__":
    run_slim_test()
