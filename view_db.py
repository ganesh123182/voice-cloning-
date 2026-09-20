"""
TrustVoice Database & Ledger Viewer
Quickly inspect all SQLite records and tamper-evident ledger entries.
Usage: python view_db.py
"""
import os
import sqlite3
import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "backend" / "voice_enrollment.db"
LEDGER_PATH = BASE_DIR / "backend" / "ledger.json"

def print_separator(title=""):
    width = 78
    if title:
        padding = (width - len(title) - 4) // 2
        print("\n" + "=" * padding + f"[ {title} ]" + "=" * (width - len(title) - 4 - padding))
    else:
        print("=" * width)

def view_database():
    print_separator("TRUSTVOICE DATABASE INSPECTOR")
    print(f" Database File : {DB_PATH}")
    print(f" Status        : {'EXISTS' if DB_PATH.exists() else 'NOT FOUND'}")
    if DB_PATH.exists():
        print(f" File Size     : {DB_PATH.stat().st_size:,} bytes")
        print(f" Last Modified : {datetime.fromtimestamp(DB_PATH.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")

    if not DB_PATH.exists():
        print("\n[!] Database file does not exist yet. Run backend to create it.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. Inspect Users Table
    print_separator("TABLE: users")
    try:
        cursor.execute("SELECT id, username, full_name, email, phone FROM users ORDER BY rowid DESC")
        users = cursor.fetchall()
        if not users:
            print(" (No user records found)")
        else:
            print(f" Total Users: {len(users)}\n")
            print(f" {'ID':<38} | {'USERNAME':<16} | {'FULL NAME':<18} | {'EMAIL/PHONE'}")
            print("-" * 78)
            for u in users:
                uid = str(u["id"])
                uname = str(u["username"] or "N/A")[:15]
                fname = str(u["full_name"] or "N/A")[:17]
                contact = str(u["email"] or u["phone"] or "N/A")
                print(f" {uid:<38} | {uname:<16} | {fname:<18} | {contact}")
    except Exception as e:
        print(f" Error querying users: {e}")

    # 2. Inspect Voice Enrollments Table
    print_separator("TABLE: voice_enrollments")
    try:
        cursor.execute("""
            SELECT enrollment_id, user_id, enrollment_version, voice_hash, 
                   evidence_hash, audio_filepath, status, created_at 
            FROM voice_enrollments 
            ORDER BY created_at DESC
        """)
        enrolls = cursor.fetchall()
        if not enrolls:
            print(" (No voice enrollment records found)")
        else:
            print(f" Total Voice Enrollments: {len(enrolls)}\n")
            for idx, e in enumerate(enrolls, 1):
                eid = e["enrollment_id"]
                uid = e["user_id"]
                ver = e["enrollment_version"]
                v_hash = e["voice_hash"] or "N/A"
                e_hash = e["evidence_hash"] or "N/A"
                audio = e["audio_filepath"] or "N/A"
                status = e["status"]
                created = e["created_at"]
                
                audio_exists = " [EXISTS]" if audio != "N/A" and Path(audio).exists() else ""

                print(f" [{idx}] Enrollment ID : {eid}")
                print(f"     User ID       : {uid} (Version {ver})")
                print(f"     Status        : {status} | Timestamp: {created}")
                print(f"     Voice Hash    : {v_hash}")
                print(f"     Evidence Hash : {e_hash}")
                print(f"     Audio File    : {audio}{audio_exists}")
                print("-" * 78)
    except Exception as e:
        print(f" Error querying voice_enrollments: {e}")

    conn.close()

    # 3. Inspect Blockchain Ledger JSON
    print_separator("CRYPTO LEDGER: ledger.json")
    if not LEDGER_PATH.exists():
        print(" (No ledger.json file found)")
    else:
        try:
            with open(LEDGER_PATH, "r") as f:
                ledger = json.load(f)
            print(f" Total Ledger Profiles: {len(ledger)}\n")
            print(f" {'USER ID':<36} | {'VOICE HASH (SHA-256)':<34} | {'STATUS'}")
            print("-" * 78)
            for uid, data in ledger.items():
                v_hash = data.get("sha256_hash") or data.get("voice_hash") or "N/A"
                v_preview = v_hash[:28] + "..." if len(v_hash) > 31 else v_hash
                vp_len = len(data.get("voiceprint", []))
                print(f" {uid:<36} | {v_preview:<34} | {vp_len}-dim Vector")
        except Exception as e:
            print(f" Error reading ledger.json: {e}")

    # 4. Integrity Check
    print_separator("INTEGRITY VERIFICATION")
    try:
        from backend.blockchain import verify_integrity
        valid, msg = verify_integrity()
        icon = "[PASS]" if valid else "[FAIL]"
        print(f" {icon} Result: {msg}")
    except Exception as e:
        print(f" [!] Integrity check skipped: {e}")

    print_separator()
    print(" Inspection complete.\n")

if __name__ == "__main__":
    view_database()
