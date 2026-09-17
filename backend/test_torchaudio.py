import torchaudio
import os
print("Testing torchaudio")
try:
    # create a dummy file
    with open("dummy.wav", "w") as f:
        f.write("not a wav file")
    signal, fs = torchaudio.load("dummy.wav")
    print("Success")
except Exception as e:
    print(f"Error loading: {e}")
finally:
    if os.path.exists("dummy.wav"):
        os.remove("dummy.wav")
