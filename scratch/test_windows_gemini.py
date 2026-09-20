import sys
sys.path.insert(0, '.')
import librosa
import numpy as np
from backend.voice_detector import detector
from backend.audio_utils import condition_speakerphone_audio

y, sr = librosa.load('google_tts_test.mp3', sr=16000)
WINDOW_SIZE = int(sr * 3.0)
STRIDE_SIZE = int(sr * 1.5)

print(f"Total audio length: {len(y)/sr:.2f}s ({len(y)} samples)")

audio_buffer = np.zeros(0, dtype=np.float32)
audio_buffer = np.concatenate([audio_buffer, y])

window_idx = 0
while len(audio_buffer) >= WINDOW_SIZE:
    full_window = audio_buffer[:WINDOW_SIZE]
    audio_buffer = audio_buffer[STRIDE_SIZE:]
    window_idx += 1
    
    rms = float(np.sqrt(np.mean(full_window**2)))
    peak = float(np.max(np.abs(full_window)))
    
    # 1. Unconditioned
    res_uncond = detector.analyze_raw(full_window, sr=sr)
    sc_uncond = res_uncond.get('scores', {})
    
    # 2. Conditioned
    cond_win = condition_speakerphone_audio(full_window, sr=sr)
    res_cond = detector.analyze_raw(cond_win, sr=sr)
    sc_cond = res_cond.get('scores', {})
    
    print(f"Window {window_idx:2d} (rms={rms:.4f}, peak={peak:.4f}):")
    print(f"   Unconditioned -> ai={res_uncond['ai_probability']:.1f}%, base={sc_uncond.get('base_fake_prob')*100:.1f}%, voc={sc_uncond.get('vocoder_fake_prob')*100:.1f}%")
    print(f"   Conditioned   -> ai={res_cond['ai_probability']:.1f}%, base={sc_cond.get('base_fake_prob')*100:.1f}%, voc={sc_cond.get('vocoder_fake_prob')*100:.1f}%")
