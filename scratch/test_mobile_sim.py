import sys
sys.path.insert(0, '.')
import librosa
import numpy as np
import scipy.signal as sig
from backend.voice_detector import detector
from backend.audio_utils import condition_speakerphone_audio, simulate_speakerphone_rir_and_codec

y, sr = librosa.load('google_tts_test.mp3', sr=16000)

# Simulate microphone recording in room: RIR + background noise + lower gain
y_sim = simulate_speakerphone_rir_and_codec(y, sr, reverb_intensity=0.25)
# Add small room noise
noise = np.random.randn(len(y_sim)).astype(np.float32) * 0.003
y_room = y_sim * 0.15 + noise # Target RMS ~0.015-0.020 like the user's phone mic!

WINDOW_SIZE = int(sr * 3.0)
STRIDE_SIZE = int(sr * 1.5)

audio_buffer = np.concatenate([np.zeros(0, dtype=np.float32), y_room])
win_idx = 0
print(f"Simulating mobile phone microphone capture of Gemini audio (RMS ~0.015):")
while len(audio_buffer) >= WINDOW_SIZE:
    full_win = audio_buffer[:WINDOW_SIZE]
    audio_buffer = audio_buffer[STRIDE_SIZE:]
    win_idx += 1
    rms = float(np.sqrt(np.mean(full_win**2)))
    peak = float(np.max(np.abs(full_win)))
    
    cond = condition_speakerphone_audio(full_win, sr=sr)
    res = detector.analyze_raw(cond, sr=sr)
    sc = res.get('scores', {})
    ai = res['ai_probability']
    base = sc.get('base_fake_prob', 0) * 100
    voc = sc.get('vocoder_fake_prob', 0) * 100
    print(f"Win {win_idx:2d} (rms={rms:.4f}, peak={peak:.4f}) -> ai_prob={ai:.1f}%, base_fake={base:.1f}%, vocoder={voc:.1f}%")
