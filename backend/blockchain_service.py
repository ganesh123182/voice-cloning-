import os
import json
import hashlib
from web3 import Web3

WEB3_PROVIDER_URI = os.getenv("WEB3_PROVIDER_URI")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")
CHAIN_ID = os.getenv("CHAIN_ID", 1337) # Default Ganache/Local

def generate_canonical_evidence(enrollment_id: str, user_id: str, model_version: str, enrollment_version: int, timestamp: str) -> str:
    """
    Generates deterministic SHA-256 hash of canonical evidence.
    """
    evidence = {
        "enrollment_id": enrollment_id,
        "enrollment_version": enrollment_version,
        "model_version": model_version,
        "timestamp": timestamp,
        "user_id": user_id
    }
    # Sort keys for deterministic JSON output
    serialized = json.dumps(evidence, separators=(',', ':'), sort_keys=True)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

def is_blockchain_configured() -> bool:
    return bool(WEB3_PROVIDER_URI and PRIVATE_KEY and CONTRACT_ADDRESS)

def anchor_hash_to_blockchain(evidence_hash: str, enrollment_id: str) -> str:
    """
    Anchors the SHA-256 hash to the blockchain.
    Returns the transaction hash or raises an Exception.
    """
    if not is_blockchain_configured():
        raise RuntimeError("BLOCKCHAIN_NOT_CONFIGURED")

    w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER_URI))
    if not w3.is_connected():
        raise RuntimeError("Failed to connect to Ethereum RPC")

    # Assuming a simple contract ABI with function:
    # function storeHash(string memory enrollmentId, string memory evidenceHash)
    abi = [
        {
            "inputs": [
                {"internalType": "string", "name": "enrollmentId", "type": "string"},
                {"internalType": "string", "name": "evidenceHash", "type": "string"}
            ],
            "name": "storeHash",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        }
    ]
    
    contract = w3.eth.contract(address=w3.to_checksum_address(CONTRACT_ADDRESS), abi=abi)
    account = w3.eth.account.from_key(PRIVATE_KEY)
    
    nonce = w3.eth.get_transaction_count(account.address)
    
    tx = contract.functions.storeHash(enrollment_id, evidence_hash).build_transaction({
        'chainId': int(CHAIN_ID),
        'gas': 2000000,
        'gasPrice': w3.eth.gas_price,
        'nonce': nonce,
    })
    
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    
    return w3.to_hex(tx_hash)

def verify_hash_on_blockchain(enrollment_id: str) -> str:
    """
    Retrieves the hash from the blockchain for the given enrollment_id.
    """
    if not is_blockchain_configured():
        raise RuntimeError("BLOCKCHAIN_NOT_CONFIGURED")
        
    w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER_URI))
    abi = [
        {
            "inputs": [{"internalType": "string", "name": "enrollmentId", "type": "string"}],
            "name": "getHash",
            "outputs": [{"internalType": "string", "name": "", "type": "string"}],
            "stateMutability": "view",
            "type": "function"
        }
    ]
    contract = w3.eth.contract(address=w3.to_checksum_address(CONTRACT_ADDRESS), abi=abi)
    return contract.functions.getHash(enrollment_id).call()
