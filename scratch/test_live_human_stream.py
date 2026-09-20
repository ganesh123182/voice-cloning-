import sys
sys.path.insert(0, '.')
import time
import json
import librosa
import numpy as np
import websockets
import asyncio
from backend.audio_utils import simulate_speakerphone_rir_and_codec, condition_speakerphone_audio

async def test_live_server_human():
    uri = "ws://127.0.0.1:8000/api/monitoring/live?token=test_token&caller_number=Simulation&caller_name=Simulation%20Test&is_simulation=true"
    print(f"Connecting to {uri}...")
    
    y, sr = librosa.load("backend/storage/enrolled_voices/72e8c2a7-422f-4974-bab0-1119c544e39b.wav", sr=16000)
    # Mobile mic simulation
    y_sim = simulate_speakerphone_rir_and_codec(y, sr, reverb_intensity=0.25)
    noise = np.random.randn(len(y_sim)).astype(np.float32) * 0.003
    y_room = (y_sim * 0.15) + noise
    
    CHUNK_SAMPLES = int(sr * 1.5)
    
    async with websockets.connect(uri) as ws:
        print("[CONNECTED] Streaming simulated mobile phone capture of Real Human voice...")
        
        for i in range(0, len(y_room) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
            chunk = y_room[i:i+CHUNK_SAMPLES]
            pcm_bytes = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            await ws.send(pcm_bytes)
            
            resp_raw = await ws.recv()
            resp = json.loads(resp_raw)
            print(f"Chunk {i//CHUNK_SAMPLES + 1:2d} -> Speaker={resp.get('speaker').upper():7s} | Risk={resp.get('risk_score'):5.1f}% | Label='{resp.get('label')}' | Alert={resp.get('is_alert')}")

asyncio.run(test_live_server_human())
