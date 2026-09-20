import sys
sys.path.insert(0, '.')
import librosa
import numpy as np
from backend.voice_detector import detector
from backend.audio_utils import condition_speakerphone_audio

for fname in ['edge_tts_test.mp3', 'test.wav', 'voice1.wav', 'voice2.wav']:
    y, sr = librosa.load(fname, sr=16000)
    WINDOW_SIZE = int(sr * 3.0)
    STRIDE_SIZE = int(sr * 1.5)
    print(f"\n=== Testing {fname} ({len(y)/sr:.2f}s) ===")
    
    audio_buffer = np.concatenate([np.zeros(0, dtype=np.float32), y])
    window_idx = 0
    while len(audio_buffer) >= WINDOW_SIZE:
        full_window = audio_buffer[:WINDOW_SIZE]
        audio_buffer = audio_buffer[STRIDE_SIZE:]
        window_idx += 1
        
        rms = float(np.sqrt(np.mean(full_window**2)))
        peak = float(np.max(np.abs(full_window)))
        
        res_uncond = detector.analyze_raw(full_window, sr=sr)
        sc_uncond = res_uncond.get('scores', {})
        
        cond_win = condition_speakerphone_audio(full_window, sr=sr)
        res_cond = detector.analyze_raw(cond_win, sr=sr)
        sc_cond = res_cond.get('scores', {})
        
        print(f"Win {window_idx:2d} (rms={rms:.4f}, peak={peak:.4f}):")
        print(f"   Uncond -> ai={res_uncond['ai_probability']:.1f}%, base={sc_uncond.get('base_fake_prob')*100:.1f}%, voc={sc_uncond.get('vocoder_fake_prob')*100:.1f}%")
        print(f"   Cond   -> ai={res_cond['ai_probability']:.1f}%, base={sc_cond.get('base_fake_prob')*100:.1f}%, voc={sc_cond.get('vocoder_fake_prob')*100:.1f}%")
