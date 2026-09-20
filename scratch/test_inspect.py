import sys, os
import glob
sys.path.insert(0, '.')
from backend.voice_detector import detector

print("--- TESTING TTS / SYNTHETIC SAMPLES ---")
for f in sorted(glob.glob('backend/models/tts_cache/*.mp3'))[:15]:
    res = detector.analyze(f)
    sc = res.get('scores', {})
    print(f"{os.path.basename(f):25s} -> ai_prob={res.get('ai_probability', 0):.1f}%, base={sc.get('base_fake_prob', 0)*100:.1f}%, voc={sc.get('vocoder_fake_prob', 0)*100:.1f}%, pred={res.get('prediction')}")

print("\n--- TESTING REAL SAMPLES ---")
for f in sorted(glob.glob('backend/storage/enrolled_voices/*.wav'))[:5]:
    res = detector.analyze(f)
    sc = res.get('scores', {})
    print(f"{os.path.basename(f):25s} -> ai_prob={res.get('ai_probability', 0):.1f}%, base={sc.get('base_fake_prob', 0)*100:.1f}%, voc={sc.get('vocoder_fake_prob', 0)*100:.1f}%, pred={res.get('prediction')}")

print("\n--- TESTING ROOT SAMPLES ---")
for f in ['google_tts_test.mp3', 'edge_tts_test.mp3', 'test.wav', 'test_enroll.wav', 'voice1.wav', 'voice2.wav']:
    if os.path.exists(f):
        res = detector.analyze(f)
        sc = res.get('scores', {})
        print(f"{os.path.basename(f):25s} -> ai_prob={res.get('ai_probability', 0):.1f}%, base={sc.get('base_fake_prob', 0)*100:.1f}%, voc={sc.get('vocoder_fake_prob', 0)*100:.1f}%, pred={res.get('prediction')}")
