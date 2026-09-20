import sys
sys.path.insert(0, '.')
import librosa
from backend.voice_detector import detector
from backend.audio_utils import condition_speakerphone_audio, simulate_speakerphone_rir_and_codec

files = ['google_tts_test.mp3', 'edge_tts_test.mp3', 'test.wav', 'voice1.wav', 'backend/storage/enrolled_voices/72e8c2a7-422f-4974-bab0-1119c544e39b.wav']

for f in files:
    y, sr = librosa.load(f, sr=16000)
    # 1. Raw
    res_raw = detector.analyze_raw(y, sr)
    sc_raw = res_raw.get('scores', {})
    
    # 2. Conditioned with condition_speakerphone_audio
    y_cond = condition_speakerphone_audio(y, sr)
    res_cond = detector.analyze_raw(y_cond, sr)
    sc_cond = res_cond.get('scores', {})
    
    # 3. Speakerphone simulation (RIR + reverb) THEN conditioned
    y_sim = simulate_speakerphone_rir_and_codec(y, sr)
    y_sim_cond = condition_speakerphone_audio(y_sim, sr)
    res_sim = detector.analyze_raw(y_sim_cond, sr)
    sc_sim = res_sim.get('scores', {})

    print(f"File: {f}")
    print(f"  Raw:        ai_prob={res_raw['ai_probability']:.1f}%, base={sc_raw.get('base_fake_prob', 0)*100:.1f}%, voc={sc_raw.get('vocoder_fake_prob', 0)*100:.1f}%")
    print(f"  Cond:       ai_prob={res_cond['ai_probability']:.1f}%, base={sc_cond.get('base_fake_prob', 0)*100:.1f}%, voc={sc_cond.get('vocoder_fake_prob', 0)*100:.1f}%")
    print(f"  Sim+Cond:   ai_prob={res_sim['ai_probability']:.1f}%, base={sc_sim.get('base_fake_prob', 0)*100:.1f}%, voc={sc_sim.get('vocoder_fake_prob', 0)*100:.1f}%")
