import sys, os, json, librosa, numpy as np, websockets, asyncio
sys.path.insert(0, '.')
from backend.audio_utils import simulate_speakerphone_rir_and_codec

async def test():
    real_f = r'E:\VoiceDeepfakeAI\voice_deepfake_dataset\train\real\hinglish\real_hing_Devesh Koshta_93c2c1b9fa.wav'
    y, sr = librosa.load(real_f, sr=16000)
    
    # Simulate mobile mic recording (RIR + small noise, RMS ~0.02)
    y_sim = simulate_speakerphone_rir_and_codec(y, sr=16000, reverb_intensity=0.20)
    noise = np.random.randn(len(y_sim)).astype(np.float32) * 0.003
    y_room = (y_sim * 0.20) + noise
    
    CHUNK_SAMPLES = int(sr * 1.5)
    uri = 'ws://127.0.0.1:8000/api/monitoring/live?token=test_token&caller_number=Simulation&caller_name=Simulation%20Test&is_simulation=true'
    async with websockets.connect(uri) as ws:
        print("[CONNECTED] Streaming natural continuous human speech (Devesh Koshta)...")
        for i in range(0, len(y_room) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
            chunk = y_room[i:i+CHUNK_SAMPLES]
            pcm_bytes = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            await ws.send(pcm_bytes)
            resp = json.loads(await ws.recv())
            print(f"Chunk {i//CHUNK_SAMPLES + 1} -> Speaker={resp.get('speaker').upper():7s} | Risk={resp.get('risk_score'):5.1f}% | Label='{resp.get('label')}' | Alert={resp.get('is_alert')}")

asyncio.run(test())
