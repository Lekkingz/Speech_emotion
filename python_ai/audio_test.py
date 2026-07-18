from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# LOAD AUDIO FILE
# ==========================================

ROOT_DIR = Path(__file__).resolve().parents[1]
audio_path = ROOT_DIR / "datasets" / "raw" / "Audio_Speech_Actors_01-24" / "Actor_01" / "03-01-01-01-01-01-01.wav"

# y = audio signal
# sr = sample rate
y, sr = librosa.load(audio_path, sr=None)

print("Audio Loaded Successfully!")
print("Sample Rate:", sr)
print("Audio Shape:", y.shape)

# ==========================================
# PLOT WAVEFORM
# ==========================================

plt.figure(figsize=(12, 4))

librosa.display.waveshow(y, sr=sr)

plt.title("Waveform")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.show()

# ==========================================
# EXTRACT MFCC FEATURES
# ==========================================

mfcc = librosa.feature.mfcc(
    y=y,
    sr=sr,
    n_mfcc=13
)

print("\nMFCC Shape:", mfcc.shape)

# ==========================================
# DISPLAY MFCC
# ==========================================

plt.figure(figsize=(12, 4))

librosa.display.specshow(
    mfcc,
    x_axis='time'
)

plt.colorbar()

plt.title("MFCC Features")

plt.tight_layout()

plt.show()