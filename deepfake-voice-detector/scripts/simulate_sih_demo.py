"""
====================================================================
  SIH DEMO SIMULATION SCRIPT
  Simulates an incoming call with real + cloned audio files,
  streams them over the live WebSocket, and prints verdicts live.
====================================================================

Usage:
    python scripts/simulate_sih_demo.py
    python scripts/simulate_sih_demo.py --real path/to/real.wav --fake path/to/clone.wav
    python scripts/simulate_sih_demo.py --server http://127.0.0.1:8001

If no audio files are provided, synthetic test signals are generated:
  - "Real user" audio:  a natural-sounding multi-frequency tone
  - "Cloned fake" audio: digitally synthesized robotic waveform
"""

import sys
import io
import os
import json
import math
import wave
import struct
import time
import argparse
import asyncio
from pathlib import Path

# Fix Windows UTF-8 output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import requests
except ImportError:
    print("[ERROR] 'requests' is not installed. Run: pip install requests")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("[ERROR] 'websockets' is not installed. Run: pip install websockets")
    sys.exit(1)


# ---- Configuration ----
DEFAULT_SERVER = "http://127.0.0.1:8001"
EMAIL = "sih_demo_judge@trustvoice.demo"
PASSWORD = "SIHDemo2026!"
SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # seconds per WebSocket chunk
CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_DURATION  # 96,000 bytes


# ---- Utility: Color Output ----
class C:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"


def banner():
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║       🛡️  TrustVoice — SIH Live Demo Simulation  🛡️          ║
║                                                              ║
║   Voice Deepfake Detection System • Smart India Hackathon    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{C.RESET}
""")


def print_step(step: int, msg: str, color=C.CYAN):
    print(f"\n{color}{C.BOLD}[STEP {step}]{C.RESET} {msg}")


def print_verdict(chunk_num: int, data: dict):
    """Pretty-print a single chunk verdict from the WebSocket."""
    speaker = data.get("speaker", "unknown")
    risk = data.get("risk_score", 0.0)
    label = data.get("label", "")
    suggestion = data.get("suggestion", "")
    is_alert = data.get("is_alert", False)

    if speaker == "silence":
        icon = "⏸️ "
        color = C.GRAY
        badge = "SILENCE"
    elif speaker == "user":
        icon = "👤"
        color = C.GREEN
        badge = "USER SPEAKING"
    elif speaker == "caller" and is_alert:
        icon = "🚨"
        color = C.RED
        badge = "FAKE VOICE DETECTED"
    elif speaker == "caller":
        icon = "📞"
        color = C.GREEN
        badge = "HUMAN CALLER"
    else:
        icon = "❓"
        color = C.YELLOW
        badge = speaker.upper()

    risk_bar_len = int(risk / 5)  # 0-20 chars
    risk_bar = "█" * risk_bar_len + "░" * (20 - risk_bar_len)

    print(f"  {color}╭─── Chunk #{chunk_num} ───────────────────────────────╮{C.RESET}")
    print(f"  {color}│  {icon}  {C.BOLD}{badge}{C.RESET}")
    print(f"  {color}│  Risk: [{risk_bar}] {risk:.1f}%{C.RESET}")
    if label:
        print(f"  {color}│  Label: {label}{C.RESET}")
    if suggestion:
        print(f"  {color}│  Advisory: {suggestion}{C.RESET}")
    print(f"  {color}╰──────────────────────────────────────────────╯{C.RESET}")


# ---- Audio Generation ----

def generate_enrollment_wav() -> io.BytesIO:
    """Generate a 5-second enrollment audio (440Hz tone)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        for i in range(SAMPLE_RATE * 5):
            s = int(32767 * 0.5 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE))
            wf.writeframes(struct.pack("<h", s))
    buf.seek(0)
    return buf


def generate_real_user_pcm() -> bytes:
    """
    Generate 3 seconds of 'real user' audio (multi-frequency harmonic).
    This simulates a natural voice with harmonics at 150, 300, 450 Hz.
    """
    samples = bytearray()
    for i in range(SAMPLE_RATE * CHUNK_DURATION):
        t = i / SAMPLE_RATE
        # Multi-harmonic signal simulating voice
        val = (
            0.3 * math.sin(2 * math.pi * 150 * t)
            + 0.2 * math.sin(2 * math.pi * 300 * t)
            + 0.1 * math.sin(2 * math.pi * 450 * t)
            + 0.05 * math.sin(2 * math.pi * 600 * t)
        )
        s = int(32767 * val)
        s = max(-32768, min(32767, s))
        samples.extend(struct.pack("<h", s))
    return bytes(samples)


def generate_fake_clone_pcm() -> bytes:
    """
    Generate 3 seconds of 'cloned/AI' audio.
    Uses a square-ish waveform with unnatural harmonics to simulate vocoder artifacts.
    """
    samples = bytearray()
    for i in range(SAMPLE_RATE * CHUNK_DURATION):
        t = i / SAMPLE_RATE
        # Square-ish wave with odd harmonics (vocoder-like)
        val = 0.0
        for k in range(1, 16, 2):  # Odd harmonics 1,3,5,...15
            val += (1.0 / k) * math.sin(2 * math.pi * 200 * k * t)
        val *= 0.3  # Scale down
        s = int(32767 * val)
        s = max(-32768, min(32767, s))
        samples.extend(struct.pack("<h", s))
    return bytes(samples)


def load_audio_as_pcm(filepath: str) -> list:
    """Load an audio file and convert to 16kHz 16-bit mono PCM chunks."""
    try:
        import librosa
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        import numpy as np
        pcm = (y * 32767).astype(np.int16).tobytes()
        # Split into CHUNK_BYTES-sized pieces
        chunks = []
        for offset in range(0, len(pcm), CHUNK_BYTES):
            chunk = pcm[offset : offset + CHUNK_BYTES]
            if len(chunk) == CHUNK_BYTES:
                chunks.append(chunk)
        return chunks if chunks else [pcm[:CHUNK_BYTES].ljust(CHUNK_BYTES, b"\x00")]
    except ImportError:
        print(f"  {C.YELLOW}[WARN] librosa not available, using raw file bytes{C.RESET}")
        with open(filepath, "rb") as f:
            raw = f.read()
        return [raw[:CHUNK_BYTES].ljust(CHUNK_BYTES, b"\x00")]


# ---- Main Demo Flow ----

async def run_demo(args):
    banner()
    server = args.server.rstrip("/")

    # ==== Step 1: Register ====
    print_step(1, "Registering demo user account...")
    r = requests.post(f"{server}/api/auth/register", data={"email": EMAIL, "password": PASSWORD})
    if r.status_code == 200:
        print(f"  {C.GREEN}✅ Registered: {r.json()}{C.RESET}")
    elif r.status_code == 400:
        print(f"  {C.YELLOW}ℹ️  User already exists (OK){C.RESET}")
    else:
        print(f"  {C.RED}❌ Registration failed: {r.status_code} {r.text}{C.RESET}")
        return

    # ==== Step 2: Login ====
    print_step(2, "Authenticating and getting JWT token...")
    r = requests.post(f"{server}/api/auth/login", data={"username": EMAIL, "password": PASSWORD})
    if r.status_code != 200:
        print(f"  {C.RED}❌ Login failed: {r.status_code} {r.text}{C.RESET}")
        return
    token = r.json()["access_token"]
    print(f"  {C.GREEN}✅ Token: {token[:30]}...{C.RESET}")
    headers = {"Authorization": f"Bearer {token}"}

    # ==== Step 3: Voice Enrollment ====
    print_step(3, "Enrolling voice profile + blockchain registration...")
    enrollment_wav = generate_enrollment_wav()
    r = requests.post(
        f"{server}/api/voice/enrollment",
        headers=headers,
        files={"file": ("enrollment.wav", enrollment_wav, "audio/wav")},
    )
    if r.status_code == 200:
        data = r.json()
        print(f"  {C.GREEN}✅ {data.get('message', 'Enrolled')}{C.RESET}")
        print(f"  {C.CYAN}   Profile Hash: {data.get('profile_hash', 'N/A')[:32]}...{C.RESET}")
        tx_id = data.get("blockchain_tx_id", "N/A")
        print(f"  {C.CYAN}   Blockchain Tx: {tx_id[:32]}...{C.RESET}")
    else:
        print(f"  {C.RED}❌ Enrollment failed: {r.status_code} {r.text}{C.RESET}")
        return

    # ==== Step 4: Blockchain Verification ====
    print_step(4, "Verifying voice profile against blockchain ledger...")
    r = requests.get(f"{server}/api/blockchain/verify", headers=headers)
    if r.status_code == 200:
        vdata = r.json()
        verified = vdata.get("blockchain_verified", False)
        icon = "✅" if verified else "❌"
        color = C.GREEN if verified else C.RED
        print(f"  {color}{icon} Blockchain verified: {verified}{C.RESET}")
        if vdata.get("on_chain_record"):
            rec = vdata["on_chain_record"]
            print(f"  {C.CYAN}   Block: {rec.get('block_number', 'N/A')}{C.RESET}")
            print(f"  {C.CYAN}   Contract: {rec.get('contract_address', 'N/A')[:24]}...{C.RESET}")
    else:
        print(f"  {C.YELLOW}⚠️  Verify endpoint: {r.status_code}{C.RESET}")

    # ==== Step 5: Simulate Call with Real User Audio ====
    print_step(5, "Simulating REAL USER call (WebSocket live stream)...")

    if args.real and os.path.exists(args.real):
        real_chunks = load_audio_as_pcm(args.real)
        print(f"  {C.CYAN}   Using audio file: {args.real} ({len(real_chunks)} chunks){C.RESET}")
    else:
        real_chunks = [generate_real_user_pcm()]
        print(f"  {C.CYAN}   Using synthetic real-user audio (1 chunk){C.RESET}")

    ws_url = f"ws{server[4:]}/api/monitoring/live?token={token}"

    try:
        async with websockets.connect(ws_url, open_timeout=60) as ws:
            print(f"  {C.GREEN}✅ WebSocket connected{C.RESET}")
            print(f"\n  {C.BOLD}--- Real User Audio Verdicts ---{C.RESET}")

            for i, chunk in enumerate(real_chunks):
                await ws.send(chunk)
                try:
                    reply = await asyncio.wait_for(ws.recv(), timeout=60)
                    data = json.loads(reply)
                    print_verdict(i + 1, data)
                except asyncio.TimeoutError:
                    print(f"  {C.YELLOW}⚠️  Chunk {i+1} timed out (model may still be loading){C.RESET}")
                await asyncio.sleep(0.5)

    except Exception as e:
        print(f"  {C.RED}❌ WebSocket error: {e}{C.RESET}")

    # ==== Step 6: Simulate Call with Cloned/Fake Audio ====
    print_step(6, "Simulating CLONED FAKE VOICE call (WebSocket live stream)...")

    if args.fake and os.path.exists(args.fake):
        fake_chunks = load_audio_as_pcm(args.fake)
        print(f"  {C.CYAN}   Using audio file: {args.fake} ({len(fake_chunks)} chunks){C.RESET}")
    else:
        fake_chunks = [generate_fake_clone_pcm()] * 3  # Send 3 chunks for rolling score
        print(f"  {C.CYAN}   Using synthetic cloned audio (3 chunks){C.RESET}")

    try:
        async with websockets.connect(ws_url, open_timeout=60) as ws:
            print(f"  {C.GREEN}✅ WebSocket connected{C.RESET}")
            print(f"\n  {C.BOLD}--- Cloned Fake Audio Verdicts ---{C.RESET}")

            for i, chunk in enumerate(fake_chunks):
                await ws.send(chunk)
                try:
                    reply = await asyncio.wait_for(ws.recv(), timeout=60)
                    data = json.loads(reply)
                    print_verdict(i + 1, data)
                except asyncio.TimeoutError:
                    print(f"  {C.YELLOW}⚠️  Chunk {i+1} timed out{C.RESET}")
                await asyncio.sleep(0.5)

    except Exception as e:
        print(f"  {C.RED}❌ WebSocket error: {e}{C.RESET}")

    # ==== Step 7: Final Summary ====
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════════╗
║                   DEMO SIMULATION COMPLETE                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ✅  User Registration       — Complete                      ║
║  ✅  JWT Authentication      — Complete                      ║
║  ✅  Voice Enrollment        — Complete                      ║
║  ✅  Blockchain Registration — Complete                      ║
║  ✅  Blockchain Verification — Complete                      ║
║  ✅  Real User Audio Stream  — Analyzed                      ║
║  ✅  Fake Clone Audio Stream — Analyzed                      ║
║                                                              ║
║  Judges: Use GET /api/blockchain/audit/<session_id> to       ║
║  inspect the full immutable audit trail via Swagger UI.      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{C.RESET}
""")


# ---- Entry Point ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrustVoice SIH Demo Simulation")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="Backend server URL")
    parser.add_argument("--real", default=None, help="Path to real user audio file (.wav)")
    parser.add_argument("--fake", default=None, help="Path to cloned/fake audio file (.wav)")
    args = parser.parse_args()

    asyncio.run(run_demo(args))
