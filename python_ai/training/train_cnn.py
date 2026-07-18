"""Train a CNN-based emotion classifier on the provided dataset.

This script extracts MFCC-based 2D inputs, builds a configurable CNN,
and saves the trained model and label encoder under `models/trained/`.

Edit the `CONFIG` section below to fill in hyperparameters and architecture
before running, or pass overrides on the command line.
"""
from pathlib import Path
import argparse
import json
import os
from typing import List

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, callbacks, utils


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "datasets" / "raw" / "Audio_Speech_Actors_01-24"
MODEL_DIR = ROOT / "models" / "trained"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Default configurable values
# ----------------------------
CONFIG = {
    "n_mfcc": 40,
    "max_pad_len": 174,  # time frames (adjust based on dataset/record length)
    "input_shape": None,  # will be (n_mfcc, max_pad_len, 1)
    "cnn_filters": [32, 64],
    "kernel_size": (3, 3),
    "pool_size": (2, 2),
    "dropout_rate": 0.3,
    "dense_units": 128,
    "optimizer": "adam",
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 30,
    "test_size": 0.2,
}


def extract_mfcc(path: Path, n_mfcc: int, max_pad_len: int) -> np.ndarray:
    y, sr = librosa.load(str(path), sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # pad / truncate
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc


def load_dataset(dataset_path: Path, n_mfcc: int, max_pad_len: int):
    X = []
    y = []
    emotion_map = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    for root, _, files in os.walk(dataset_path):
        for fname in files:
            if not fname.lower().endswith('.wav'):
                continue
            try:
                code = fname.split('-')[2]
            except Exception:
                continue
            label = emotion_map.get(code, None)
            if label is None:
                continue

            file_path = Path(root) / fname
            mfcc = extract_mfcc(file_path, n_mfcc=n_mfcc, max_pad_len=max_pad_len)
            X.append(mfcc)
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    return X, y


def build_model(input_shape, num_classes, cfg: dict):
    inp = layers.Input(shape=input_shape)
    x = inp
    for f in cfg['cnn_filters']:
        x = layers.Conv2D(f, cfg['kernel_size'], activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(cfg['pool_size'])(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(cfg['dropout_rate'])(x)
    x = layers.Dense(cfg['dense_units'], activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inp, out)

    # compile
    opt_name = cfg.get('optimizer', 'adam')
    if opt_name == 'adam':
        from tensorflow.keras.optimizers import Adam
        opt = Adam(learning_rate=cfg.get('learning_rate', 0.001))
    else:
        from tensorflow.keras.optimizers import Adam
        opt = Adam(learning_rate=cfg.get('learning_rate', 0.001))

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main(args):
    cfg = CONFIG.copy()
    cfg.update({k: v for k, v in vars(args).items() if v is not None})

    n_mfcc = cfg['n_mfcc']
    max_pad_len = cfg['max_pad_len']

    print('Loading dataset...')
    X, y = load_dataset(DATASET_PATH, n_mfcc=n_mfcc, max_pad_len=max_pad_len)
    print('Loaded', X.shape, 'labels', np.unique(y))

    # reshape for conv2d: (samples, n_mfcc, time, 1)
    X = X[..., np.newaxis]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = utils.to_categorical(y_enc)

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=cfg['test_size'], random_state=42, stratify=y_enc)

    input_shape = X_train.shape[1:]
    cfg['input_shape'] = input_shape

    model = build_model(input_shape, y_cat.shape[1], cfg)
    model.summary()

    cb = [
        callbacks.ModelCheckpoint(str(MODEL_DIR / 'cnn_emotion_model.h5'), save_best_only=True, monitor='val_accuracy'),
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ]

    hist = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=cfg['epochs'],
        batch_size=cfg['batch_size'],
        callbacks=cb,
    )

    # save label encoder and config
    import joblib
    joblib.dump(le, MODEL_DIR / 'cnn_label_encoder.pkl')
    with open(MODEL_DIR / 'cnn_config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    # Save training history
    np.save(MODEL_DIR / 'cnn_history.npy', hist.history)

    print('Training finished. Model saved to', MODEL_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_mfcc', type=int, default=CONFIG['n_mfcc'])
    parser.add_argument('--max_pad_len', type=int, default=CONFIG['max_pad_len'])
    parser.add_argument('--batch_size', type=int, default=CONFIG['batch_size'])
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'])
    parser.add_argument('--dropout_rate', type=float, default=CONFIG['dropout_rate'])
    parser.add_argument('--dense_units', type=int, default=CONFIG['dense_units'])
    parser.add_argument('--test_size', type=float, default=CONFIG['test_size'])
    parser.add_argument('--learning_rate', type=float, default=CONFIG['learning_rate'])
    args = parser.parse_args()
    main(args)
