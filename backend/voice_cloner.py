"""
Voice Cloning Module - Handles text-to-speech with voice cloning.

Supports two modes:
1. REAL MODE: Uses a pretrained TTS model (when available and compatible)
2. DEMO MODE: Generates a synthetic waveform for UI demonstration
"""
import os
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

from backend.config import DEMO_MODE, GENERATED_DIR, UPLOAD_DIR
from backend.audio_utils import (
    write_wav, read_wav, generate_demo_speech, convert_to_wav, get_audio_duration
)


class VoiceCloner:
    """
    Voice cloning engine that generates speech from text using a reference voice.
    
    In Demo Mode: Generates a simple synthetic waveform (clearly labelled).
    In Real Mode: Would use XTTS/OpenVoice/F5-TTS (requires CUDA + compatible Python).
    """

    def __init__(self):
        self.demo_mode = DEMO_MODE
        self.model = None
        self.model_name = "Demo Synthesizer"

        if not self.demo_mode:
            self._try_load_model()
        else:
            print("[INFO] Voice Cloning running in DEMO MODE")
            print("   Real TTS models require CUDA GPU + Python 3.10/3.11")
            print("   Demo mode generates synthetic audio for UI demonstration")

    def _try_load_model(self):
        """Attempt to load a real TTS model."""
        # Try TTS (Coqui) with XTTS
        try:
            # pyrefly: ignore [missing-import]
            from TTS.api import TTS
            import torch

            if not torch.cuda.is_available():
                print("[WARN] CUDA not available. Falling back to Demo Mode.")
                self.demo_mode = True
                return

            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
            self.model_name = "XTTS v2"
            self.demo_mode = False
            print("[OK] Loaded XTTS v2 model with CUDA")
            return
        except ImportError:
            print("[INFO] TTS (Coqui) not installed")
        except Exception as e:
            print(f"[WARN] Could not load XTTS: {e}")

        # If no model loaded, use demo mode
        self.demo_mode = True
        print("[INFO] Falling back to DEMO MODE for voice cloning")

    def clone_voice(
        self,
        text: str,
        reference_audio_path: str,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Generate speech from text using the reference voice.
        
        Args:
            text: Text to convert to speech
            reference_audio_path: Path to the reference voice audio file
            language: Language code (default: "en")
            
        Returns:
            Dict with:
            - output_path: Path to generated audio file
            - duration: Duration in seconds
            - mode: "demo" or "real"
            - model: Name of model used
        """
        output_filename = f"cloned_{uuid.uuid4().hex[:12]}.wav"
        output_path = str(GENERATED_DIR / output_filename)

        # Ensure reference is WAV
        ref_wav_path = reference_audio_path
        if not reference_audio_path.lower().endswith('.wav'):
            ref_wav_path = reference_audio_path + ".ref.wav"
            try:
                ref_wav_path = convert_to_wav(reference_audio_path, ref_wav_path)
            except Exception:
                ref_wav_path = reference_audio_path

        if self.demo_mode:
            return self._generate_demo(text, ref_wav_path, output_path)
        else:
            return self._generate_real(text, ref_wav_path, output_path, language)

    def _generate_demo(self, text: str, ref_path: str, output_path: str) -> Dict[str, Any]:
        """Generate demo audio (clearly labelled as synthetic demo)."""
        samples, sample_rate = generate_demo_speech(text, ref_path)
        write_wav(output_path, samples, sample_rate)

        duration = len(samples) / sample_rate

        return {
            "output_path": output_path,
            "output_filename": os.path.basename(output_path),
            "duration": round(duration, 2),
            "mode": "demo",
            "model": "Demo Synthesizer",
            "disclaimer": (
                "[DEMO MODE] This audio was generated using a simple synthesizer, "
                "NOT a real voice cloning model. Real voice cloning requires "
                "CUDA GPU + compatible TTS models (XTTS/OpenVoice/F5-TTS)."
            )
        }

    def _generate_real(self, text: str, ref_path: str, output_path: str, language: str) -> Dict[str, Any]:
        """Generate real cloned voice using the loaded TTS model."""
        try:
            self.model.tts_to_file(
                text=text,
                speaker_wav=ref_path,
                language=language,
                file_path=output_path
            )

            duration = get_audio_duration(output_path)

            return {
                "output_path": output_path,
                "output_filename": os.path.basename(output_path),
                "duration": round(duration, 2),
                "mode": "real",
                "model": self.model_name,
                "disclaimer": (
                    "This audio was generated using AI voice cloning. "
                    "It is a synthetic reproduction, not the original speaker."
                )
            }
        except Exception as e:
            # Fallback to demo mode on error
            print(f"[WARN] Real TTS failed: {e}. Falling back to demo.")
            return self._generate_demo(text, ref_path, output_path)

    def get_status(self) -> Dict[str, Any]:
        """Return the current status of the voice cloner."""
        return {
            "demo_mode": self.demo_mode,
            "model_name": self.model_name,
            "model_loaded": self.model is not None,
            "message": (
                "Running in DEMO MODE - synthetic audio only"
                if self.demo_mode
                else f"Real model loaded: {self.model_name}"
            )
        }


# Singleton instance
cloner = VoiceCloner()
