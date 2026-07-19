"""Train and evaluate the Chapter Four CNN/CRNN comparison models.

This script creates the missing models needed for Tables 4.2 and 4.3:

- CNN + LPC
- CNN + LPC + MFCC
- CRNN + MFCC
- CRNN + LPC + MFCC

It also evaluates the existing CNN + MFCC model if
`models/trained/cnn_emotion_model.h5` is present.

Outputs are saved under `models/trained/chapter4/`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import callbacks, layers, models, utils
from tensorflow.keras.models import load_model


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "datasets" / "raw" / "Audio_Speech_Actors_01-24"
MODEL_DIR = ROOT / "models" / "trained"
OUT_DIR = MODEL_DIR / "chapter4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMOTION_LABELS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def dataset_files() -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    for root, dirs, files in os.walk(DATASET_PATH):
        dirs.sort()
        for fname in sorted(files):
            if not fname.lower().endswith(".wav"):
                continue
            parts = fname.split("-")
            if len(parts) < 3:
                continue
            label = EMOTION_LABELS.get(parts[2])
            if label:
                pairs.append((Path(root) / fname, label))
    return pairs


def pad_or_trim_matrix(x: np.ndarray, width: int) -> np.ndarray:
    if x.shape[1] < width:
        return np.pad(x, ((0, 0), (0, width - x.shape[1])), mode="constant")
    return x[:, :width]


def lpc_coefficients(audio: np.ndarray, order: int = 13) -> np.ndarray:
    try:
        coeffs = librosa.lpc(audio, order=order)
    except Exception:
        coeffs = np.zeros(order + 1, dtype=np.float32)
    coeffs = np.asarray(coeffs, dtype=np.float32)
    if coeffs.size < order + 1:
        coeffs = np.pad(coeffs, (0, order + 1 - coeffs.size), mode="constant")
    return coeffs[: order + 1]


def load_feature_sets(n_mfcc: int = 40, max_pad_len: int = 174, lpc_order: int = 13):
    pairs = dataset_files()
    labels = []
    mfcc_images = []
    lpc_vectors = []
    hybrid_vectors = []
    mfcc_sequences = []
    hybrid_sequences = []

    for path, label in pairs:
        audio, sr = librosa.load(str(path), sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc = pad_or_trim_matrix(mfcc, max_pad_len)
        lpc = lpc_coefficients(audio, order=lpc_order)

        mfcc_mean = np.mean(mfcc.T, axis=0)
        mfcc_std = np.std(mfcc.T, axis=0)
        hybrid = np.concatenate((mfcc_mean, mfcc_std, lpc)).astype(np.float32)

        mfcc_seq = mfcc.T.astype(np.float32)
        lpc_seq = np.repeat(lpc[np.newaxis, :], max_pad_len, axis=0)
        hybrid_seq = np.concatenate((mfcc_seq, lpc_seq), axis=1).astype(np.float32)

        labels.append(label)
        mfcc_images.append(mfcc.astype(np.float32))
        lpc_vectors.append(lpc.astype(np.float32))
        hybrid_vectors.append(hybrid)
        mfcc_sequences.append(mfcc_seq)
        hybrid_sequences.append(hybrid_seq)

    return {
        "labels": np.asarray(labels),
        "mfcc_images": np.asarray(mfcc_images)[..., np.newaxis],
        "lpc_vectors": np.asarray(lpc_vectors)[..., np.newaxis],
        "hybrid_vectors": np.asarray(hybrid_vectors)[..., np.newaxis],
        "mfcc_sequences": np.asarray(mfcc_sequences),
        "hybrid_sequences": np.asarray(hybrid_sequences),
    }


def macro_specificity(cm: np.ndarray) -> tuple[float, list[float]]:
    total = cm.sum()
    values = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - (tp + fp + fn)
        values.append(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0)
    return float(np.mean(values)), values


def scale_train_test(X_train: np.ndarray, X_test: np.ndarray, scaler_path: Path):
    scaler = StandardScaler()
    original_train_shape = X_train.shape
    original_test_shape = X_test.shape
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    X_test_flat = X_test.reshape((X_test.shape[0], -1))
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(original_train_shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(original_test_shape)
    joblib.dump(scaler, scaler_path)
    return X_train_scaled, X_test_scaled


def build_cnn_1d(input_shape, num_classes: int):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, activation="relu", padding="same"),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_crnn(input_shape, num_classes: int):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 5, activation="relu", padding="same")(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(name: str, model, X_train, y_train_cat, X_test, y_test_cat, epochs: int, batch_size: int, force: bool):
    model_path = OUT_DIR / f"{name}.keras"
    history_path = OUT_DIR / f"{name}_history.npy"
    if model_path.exists() and not force:
        return load_model(model_path)

    cb = [
        callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor="val_accuracy"),
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ]
    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=2,
    )
    np.save(history_path, history.history)
    model.save(model_path)
    return model


def evaluate_model(name: str, model, X_train, X_test, y_test, classes):
    _ = model.predict(X_test[: min(8, len(X_test))], verbose=0)
    start = time.perf_counter()
    probs = model.predict(X_test, verbose=0)
    latency = (time.perf_counter() - start) / len(X_test)
    y_pred = np.argmax(probs, axis=1)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(classes))))
    spec, per_class_spec = macro_specificity(cm)
    report = classification_report(y_test, y_pred, target_names=classes, zero_division=0)

    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "specificity_macro": spec,
        "specificity_per_class": dict(zip(classes, per_class_spec)),
        "latency_sec": float(latency),
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "num_classes": int(len(classes)),
        "classes": list(classes),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def format_metric(value):
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def format_latency(value):
    if value is None:
        return "N/A"
    return f"{value:.6f}"


def write_report(results: dict):
    json_path = OUT_DIR / "chapter4_metrics.json"
    txt_path = OUT_DIR / "chapter4_tables.txt"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    cnn = results["tables"]["cnn"]
    crnn = results["tables"]["crnn"]
    lines = []
    lines.append("======================================================")
    lines.append("Table 4.2 CNN Model Comparison on RAVDESS Dataset")
    lines.append("======================================================")
    lines.append("")
    lines.append("| Model | Accuracy | Precision | Recall | F1-score | Specificity | Latency (sec) |")
    lines.append("|-------|----------|-----------|--------|----------|-------------|---------------|")
    for label in ["CNN + LPC", "CNN + MFCC", "CNN + LPC + MFCC"]:
        r = cnn[label]
        lines.append(
            f"| {label} | {format_metric(r.get('accuracy'))} | {format_metric(r.get('precision_macro'))} | "
            f"{format_metric(r.get('recall_macro'))} | {format_metric(r.get('f1_macro'))} | "
            f"{format_metric(r.get('specificity_macro'))} | "
            f"{format_latency(r.get('latency_sec'))} |"
        )
    lines.append("")
    lines.append("======================================================")
    lines.append("Table 4.3 CRNN Model Comparison on RAVDESS Dataset")
    lines.append("======================================================")
    lines.append("")
    lines.append("| Model | Accuracy | Precision | Recall | F1-score | Specificity | Latency (sec) |")
    lines.append("|-------|----------|-----------|--------|----------|-------------|---------------|")
    for label in ["CRNN + MFCC", "CRNN + LPC + MFCC"]:
        r = crnn[label]
        lines.append(
            f"| {label} | {format_metric(r.get('accuracy'))} | {format_metric(r.get('precision_macro'))} | "
            f"{format_metric(r.get('recall_macro'))} | {format_metric(r.get('f1_macro'))} | "
            f"{format_metric(r.get('specificity_macro'))} | "
            f"{format_latency(r.get('latency_sec'))} |"
        )
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, txt_path


def existing_cnn_mfcc_result(data, le, y_enc, y_cat, test_size):
    model_path = MODEL_DIR / "cnn_emotion_model.h5"
    if not model_path.exists():
        return None
    X = data["mfcc_images"]
    X_train, X_test, _, _, _, y_test = train_test_split(
        X,
        y_cat,
        y_enc,
        test_size=test_size,
        random_state=42,
        stratify=y_enc,
    )
    model = load_model(model_path)
    return evaluate_model("CNN + MFCC", model, X_train, X_test, y_test, le.classes_)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data = load_feature_sets()
    labels = data["labels"]

    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    y_cat = utils.to_categorical(y_enc)
    joblib.dump(le, OUT_DIR / "label_encoder.pkl")

    split_kwargs = {
        "test_size": args.test_size,
        "random_state": 42,
        "stratify": y_enc,
    }

    results = {
        "dataset": {
            "total_samples": int(len(labels)),
            "num_classes": int(len(le.classes_)),
            "classes": le.classes_.tolist(),
            "test_size": args.test_size,
        },
        "models": {},
        "tables": {"cnn": {}, "crnn": {}},
    }

    cnn_mfcc = existing_cnn_mfcc_result(data, le, y_enc, y_cat, args.test_size)
    if cnn_mfcc:
        results["models"]["CNN + MFCC"] = cnn_mfcc
        results["tables"]["cnn"]["CNN + MFCC"] = cnn_mfcc

    specs = [
        ("CNN + LPC", "cnn_lpc", "lpc_vectors", build_cnn_1d),
        ("CNN + LPC + MFCC", "cnn_lpc_mfcc", "hybrid_vectors", build_cnn_1d),
        ("CRNN + MFCC", "crnn_mfcc", "mfcc_sequences", build_crnn),
        ("CRNN + LPC + MFCC", "crnn_lpc_mfcc", "hybrid_sequences", build_crnn),
    ]

    for table_name, artifact_name, feature_key, builder in specs:
        X = data[feature_key]
        X_train, X_test, y_train_cat, y_test_cat, y_train, y_test = train_test_split(
            X,
            y_cat,
            y_enc,
            **split_kwargs,
        )
        X_train, X_test = scale_train_test(X_train, X_test, OUT_DIR / f"{artifact_name}_scaler.pkl")
        model = builder(X_train.shape[1:], len(le.classes_))
        model = train_model(
            artifact_name,
            model,
            X_train,
            y_train_cat,
            X_test,
            y_test_cat,
            args.epochs,
            args.batch_size,
            args.force,
        )
        result = evaluate_model(table_name, model, X_train, X_test, y_test, le.classes_)
        results["models"][table_name] = result
        table_key = "crnn" if table_name.startswith("CRNN") else "cnn"
        results["tables"][table_key][table_name] = result

    for label in ["CNN + LPC", "CNN + MFCC", "CNN + LPC + MFCC"]:
        results["tables"]["cnn"].setdefault(label, {"model": label})
    for label in ["CRNN + MFCC", "CRNN + LPC + MFCC"]:
        results["tables"]["crnn"].setdefault(label, {"model": label})

    best = max(results["models"].values(), key=lambda r: r["accuracy"])
    results["best_model"] = {
        "model": best["model"],
        "accuracy": best["accuracy"],
        "f1_macro": best["f1_macro"],
    }

    json_path, txt_path = write_report(results)
    print(json.dumps(results["best_model"], indent=2))
    print(f"Saved metrics to: {json_path}")
    print(f"Saved copy-ready tables to: {txt_path}")


if __name__ == "__main__":
    main()
