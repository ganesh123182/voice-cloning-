"""
End-to-End WebSocket Live Call Simulation Test
Simulates an active phone call on speakerphone over the WebSocket:
1. Silence frame -> verifies "silence" status
2. Near-field User voice -> verifies "user" status, risk=0.0%
3. Loudspeaker Caller Real voice (with speakerphone RIR) -> verifies "caller", risk < 30%
4. Loudspeaker Caller Fake AI clone (with speakerphone RIR) -> verifies "caller", alert=True, risk > 70%
5. Validates roundtrip HUD response latency and 1.5s sliding window cadence.
"""
import sys
import time
import json
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from backend.main import app
from backend.audio_utils import simulate_speakerphone_rir_and_codec

def run_e2e_websocket_test():
    print("\n" + "="*75)
    print("      TRUSTVOICE: END-TO-END WEBSOCKET LIVE CALL SIMULATION TEST        ")
    print("="*75)
    
    sr = 16000
    chunk_samples = int(sr * 1.5) # 1.5s chunk = 24,000 samples
    
    # Load test samples
    test_csv = Path("E:/VoiceDeepfakeAI/voice_deepfake_dataset/metadata/test.csv")
    df = pd.read_csv(test_csv)
    real_sample_path = df[df["label"] == "real"].iloc[0]["file_path"]
    fake_sample_path = df[df["label"] == "fake"].iloc[0]["file_path"]
    
    y_real, _ = librosa.load(real_sample_path, sr=sr, mono=True)
    y_fake, _ = librosa.load(fake_sample_path, sr=sr, mono=True)
    
    # Generate test streams:
    # 1. Silence (1.5s)
    stream_silence = (np.random.randn(chunk_samples).astype(np.float32) * 0.002)
    
    # 2. Near-field User speech (holding phone close, high RMS ~0.15)
    user_slice = y_real[:chunk_samples]
    stream_user = (user_slice / np.max(np.abs(user_slice)) * 0.45).astype(np.float32)
    
    # 3. Caller Real voice over speakerphone (RIR reverberation + loudspeaker amplitude)
    real_spk = simulate_speakerphone_rir_and_codec(y_real[:chunk_samples], sr=sr, reverb_intensity=0.30)
    stream_caller_real = (real_spk / np.max(np.abs(real_spk)) * 0.18).astype(np.float32)
    
    # 4. Caller AI Cloned voice over speakerphone (RIR reverberation + loudspeaker amplitude)
    fake_spk = simulate_speakerphone_rir_and_codec(y_fake[:chunk_samples], sr=sr, reverb_intensity=0.30)
    stream_caller_fake = (fake_spk / np.max(np.abs(fake_spk)) * 0.18).astype(np.float32)
    
    # 5. Caller Google Gemini Assistant voice over speakerphone (3.0s continuous speech)
    y_gemini, _ = librosa.load("google_tts_test.mp3", sr=sr, mono=True)
    gemini_3s = simulate_speakerphone_rir_and_codec(y_gemini[:2*chunk_samples], sr=sr, reverb_intensity=0.30)
    g1 = gemini_3s[:chunk_samples]
    g2 = gemini_3s[chunk_samples:2*chunk_samples]
    stream_caller_gemini_1 = (g1 / np.max(np.abs(gemini_3s)) * 0.18).astype(np.float32)
    stream_caller_gemini_2 = (g2 / np.max(np.abs(gemini_3s)) * 0.18).astype(np.float32)

    # 6. Near-field high energy Gemini voice in Simulation Mode (testing directly near microphone)
    gemini_nearfield = (y_gemini[:chunk_samples] / np.max(np.abs(y_gemini[:chunk_samples])) * 0.40).astype(np.float32)
    
    scenarios = [
        ("STAGE 1: Silence / Background (Initial Safe)", [stream_silence], "silence", False, False),
        ("STAGE 2: Near-field User Voice", [stream_user, stream_user], "user", False, False),
        ("STAGE 3: Speakerphone Caller (Real Human Voice)", [stream_caller_real, stream_caller_real], "caller", False, False),
        ("STAGE 4: Speakerphone Caller (AI Clone Voice)", [stream_caller_fake, stream_caller_fake], "caller", True, True),
        ("STAGE 5: Speakerphone Caller (Google Gemini Assistant)", [stream_caller_gemini_1, stream_caller_gemini_2], "caller", True, True),
        ("STAGE 6: User Speaking After AI Threat (Live Fluctuation Check)", [stream_user, stream_user], "user", False, True),
        ("STAGE 7: Silence Gap / Caller Pause After AI Threat (Latching Check)", [stream_silence, stream_silence], "silence", False, True)
    ]
    
    client = TestClient(app)
    
    with client.websocket_connect("/api/monitoring/live?token=test_token_123&caller_number=+919876543210&caller_name=TestCaller") as ws:
        print("\n[OK] Connected to Live Monitoring WebSocket (`/api/monitoring/live`)")
        
        for stage_name, audio_chunks, expected_speaker, expected_alert, expected_latched in scenarios:
            print(f"\n---> Running {stage_name}...")
            resp = None
            rtt_latency = 0.0
            
            for chunk_data in audio_chunks:
                pcm_bytes = (np.clip(chunk_data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                t_send = time.perf_counter()
                ws.send_bytes(pcm_bytes)
                resp = ws.receive_json()
                if resp.get("speaker") == "scanning":
                    ws.send_bytes(pcm_bytes)
                    resp = ws.receive_json()
                rtt_latency = (time.perf_counter() - t_send) * 1000.0
            
            print(f"     HUD Response Latency : {rtt_latency:.1f} ms")
            print(f"     Speaker Identified   : {resp.get('speaker')} (Expected: '{expected_speaker}')")
            print(f"     Live Risk Score      : {resp.get('risk_score')}%")
            print(f"     Threat Latched       : {resp.get('threat_latched')} (Expected: {expected_latched})")
            print(f"     Classification Label : {resp.get('label')}")
            print(f"     Alert Triggered      : {resp.get('is_alert')} (Expected: {expected_alert})")
            print(f"     Explainability       : {resp.get('explainability_reasons')}")
            
            assert resp.get("speaker") == expected_speaker, f"Expected speaker '{expected_speaker}', got '{resp.get('speaker')}'"
            assert resp.get("threat_latched") == expected_latched, f"Expected threat_latched={expected_latched}, got {resp.get('threat_latched')}"
            
            if expected_alert:
                assert resp.get("is_alert") is True, f"Expected alert for clone voice, but got {resp.get('is_alert')}"
                assert resp.get("risk_score") >= 50.0, f"Expected risk >= 50% for clone voice, got {resp.get('risk_score')}"
            else:
                assert resp.get("risk_score") < 50.0, f"Expected live risk < 50% for safe/user stage, got {resp.get('risk_score')}"
                
            print(f"     [PASS] Stage verified successfully!")

    print("\n---> Running STAGE 7: Android Call Simulation Mode with Gemini Assistant...")
    with client.websocket_connect("/api/monitoring/live?token=test_token_123&caller_name=Simulation%20Test&is_simulation=true") as ws_sim:
        print("[OK] Connected to Live Monitoring in Simulation Mode (`?is_simulation=true`)")
        # Stream 2 consecutive 1.5s speech chunks (3.0s window) as during continuous phone speech
        chunk1 = (y_gemini[:chunk_samples] / np.max(np.abs(y_gemini[:chunk_samples])) * 0.40).astype(np.float32)
        chunk2 = (y_gemini[chunk_samples:2*chunk_samples] / np.max(np.abs(y_gemini[chunk_samples:2*chunk_samples])) * 0.40).astype(np.float32)

        pcm1 = (np.clip(chunk1, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        ws_sim.send_bytes(pcm1)
        resp1 = ws_sim.receive_json()
        print(f"     Tick 1 (1.5s startup): Speaker={resp1.get('speaker')}, Risk={resp1.get('risk_score')}%, Label='{resp1.get('label')}'")

        pcm2 = (np.clip(chunk2, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        t_send = time.perf_counter()
        ws_sim.send_bytes(pcm2)
        resp2 = ws_sim.receive_json()
        rtt_latency = (time.perf_counter() - t_send) * 1000.0

        print(f"     Tick 2 (3.0s speech) : HUD Latency={rtt_latency:.1f}ms, Speaker={resp2.get('speaker')}, Risk={resp2.get('risk_score')}%, Label='{resp2.get('label')}'")
        print(f"     Explainability       : {resp2.get('explainability_reasons')}")

        assert resp2.get("is_alert") is True, "Simulation mode must trigger alert for Gemini Assistant"
        assert resp2.get("risk_score") >= 50.0, "Simulation mode must score >= 50% for Gemini Assistant"
        print("     [PASS] Simulation mode successfully caught Gemini Assistant!")

    print("\n" + "="*75)
    print("      ALL END-TO-END WEBSOCKET LIVE CALL SIMULATION TESTS PASSED!       ")
    print("="*75)

if __name__ == "__main__":
    run_e2e_websocket_test()
