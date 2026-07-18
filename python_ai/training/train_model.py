import os
from pathlib import Path

import joblib
import numpy as np
import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# =====================================================
# DATASET PATH
# =====================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT_DIR / "datasets" / "raw" / "Audio_Speech_Actors_01-24"
MODEL_DIR = ROOT_DIR / "models" / "trained"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# EMOTION LABELS
# =====================================================

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

# =====================================================
# STORAGE
# =====================================================

X = []
y = []

# =====================================================
# PROCESS DATASET
# =====================================================

print("\nProcessing Dataset...\n")

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.endswith(".wav"):

            file_path = os.path.join(root, file)

            print(f"Processing: {file}")

            # -------------------------------------------------
            # GET EMOTION LABEL
            # -------------------------------------------------

            emotion_code = file.split("-")[2]

            emotion = emotion_labels[emotion_code]

            # -------------------------------------------------
            # LOAD AUDIO
            # -------------------------------------------------

            audio, sr = librosa.load(
                file_path,
                sr=22050
            )

            # -------------------------------------------------
            # EXTRACT HYBRID MFCC + LPC FEATURES
            # -------------------------------------------------

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=40
            )

            mfcc_mean = np.mean(
                mfcc.T,
                axis=0
            )

            mfcc_std = np.std(
                mfcc.T,
                axis=0
            )

            lpc_features = librosa.lpc(audio, order=13)

            features = np.concatenate(
                (mfcc_mean, mfcc_std, lpc_features)
            )

            if features.shape[0] > 80:
                features = features[:80]
            elif features.shape[0] < 80:
                features = np.pad(features, (0, 80 - features.shape[0]), mode='constant')

            # -------------------------------------------------
            # STORE
            # -------------------------------------------------

            X.append(features)

            y.append(emotion)

# =====================================================
# CONVERT TO NUMPY
# =====================================================

X = np.array(X)

y = np.array(y)

print("\nDataset Loaded Successfully!")

print("Feature Shape:", X.shape)

print("Labels Shape:", y.shape)

# =====================================================
# NORMALIZE FEATURES
# =====================================================

scaler = StandardScaler()

X = scaler.fit_transform(X)

print("\nFeatures Normalized!")

# =====================================================
# ENCODE LABELS
# =====================================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

print("\nEmotion Labels Encoded!")

# =====================================================
# SPLIT DATASET
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

print("\nDataset Split Complete!")

print("Training Samples:", X_train.shape[0])

print("Testing Samples:", X_test.shape[0])

# =====================================================
# CREATE MODEL
# =====================================================

model = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

# =====================================================
# TRAIN MODEL
# =====================================================

print("\nTraining AI Model...\n")

model.fit(X_train, y_train)

print("\nTraining Complete!")

# =====================================================
# PREDICTIONS
# =====================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# =====================================================
# SAVE MODEL FILES
# =====================================================

joblib.dump(
    model,
    MODEL_DIR / "emotion_model.pkl"
)

joblib.dump(
    scaler,
    MODEL_DIR / "scaler.pkl"
)

joblib.dump(
    encoder,
    MODEL_DIR / "label_encoder.pkl"
)

print("\n================================")

print("MODEL SAVED SUCCESSFULLY!")

print("================================")

print("\nSaved Files:")
print("- emotion_model.pkl")
print("- scaler.pkl")
print("- label_encoder.pkl")
