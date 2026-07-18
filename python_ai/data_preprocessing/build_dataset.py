import os
from pathlib import Path

import numpy as np
import librosa

# ==========================================
# DATASET PATH
# ==========================================

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT_DIR / "datasets" / "raw" / "Audio_Speech_Actors_01-24"

# ==========================================
# EMOTION LABELS
# ==========================================

emotion_labels = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# ==========================================
# STORAGE
# ==========================================

X = []
y = []

# ==========================================
# PROCESS FILES
# ==========================================

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.endswith(".wav"):

            file_path = os.path.join(root, file)

            print(f"Processing: {file}")

            # ----------------------------------
            # EXTRACT EMOTION
            # ----------------------------------

            parts = file.split("-")

            emotion_code = parts[2]

            emotion = emotion_labels[emotion_code]

            # ----------------------------------
            # LOAD AUDIO
            # ----------------------------------

            audio, sr = librosa.load(file_path, sr=None)

            # ----------------------------------
            # EXTRACT MFCC
            # ----------------------------------

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=13
            )

            # TAKE MEAN OF MFCC
            mfcc_mean = np.mean(mfcc.T, axis=0)

            # STORE
            X.append(mfcc_mean)
            y.append(emotion)

# ==========================================
# CONVERT TO NUMPY
# ==========================================

X = np.array(X)
y = np.array(y)

# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\nDataset Created Successfully!")

print("Feature Shape:", X.shape)
print("Labels Shape:", y.shape)

print("\nExample Emotion Labels:")
print(y[:10])
