import sys, os, json, librosa, numpy as np, websockets, asyncio, pandas as pd
sys.path.insert(0, '.')
from backend.audio_utils import simulate_speakerphone_rir_and_codec

async def test():
    df = pd.read_csv('E:/VoiceDeepfakeAI/voice_deepfake_dataset/metadata/train.csv')
    real_f = df[df['label'] == 'real']['file_path'].iloc[0]
    y, sr = librosa.load(real_f, sr=16000)
    # Loop to make 8 seconds
    y = np.tile(y, int(np.ceil(8.0 / (len(y)/sr))))
    y_sim = simulate_speakerphone_rir_and_codec(y, sr=16000, reverb_intensity=0.25)
    noise = np.random.randn(len(y_sim)).astype(np.float32) * 0.003
    y_room = (y_sim * 0.20) + noise
    
    CHUNK_SAMPLES = int(sr * 1.5)
    uri = 'ws://127.0.0.1:8000/api/monitoring/live?token=test_token&caller_number=Simulation&caller_name=Simulation%20Test&is_simulation=true'
    async with websockets.connect(uri) as ws:
        for i in range(0, min(len(y_room) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES * 5), CHUNK_SAMPLES):
            chunk = y_room[i:i+CHUNK_SAMPLES]
            pcm_bytes = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            await ws.send(pcm_bytes)
            resp = json.loads(await ws.recv())
            print(f"Chunk {i//CHUNK_SAMPLES + 1} -> Speaker={resp.get('speaker').upper():7s} | Risk={resp.get('risk_score'):5.1f}% | Label='{resp.get('label')}' | Alert={resp.get('is_alert')}")

asyncio.run(test())
