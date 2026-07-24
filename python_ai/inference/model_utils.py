import io
import json
import wave
from pathlib import Path
from typing import Optional, Tuple

import joblib
import librosa
import numpy as np


MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "trained"
TARGET_SAMPLE_RATE = 22050
EXPECTED_FEATURES = 80
DEFAULT_CNN_CONFIG = {
    "n_mfcc": 40,
    "max_pad_len": 174,
}


def _load_cnn_model(model_path: Path):
    """Load the CNN model using the available Keras-compatible backend."""
    try:
        from tensorflow.keras.models import load_model as tf_load_model
        return tf_load_model(model_path, compile=False)
    except Exception as tf_exc:
        try:
            from keras.models import load_model as keras_load_model
            return keras_load_model(model_path, compile=False)
        except Exception as keras_exc:  # pragma: no cover - runtime dependency handling
            raise ImportError(
                "Unable to load the CNN model with TensorFlow/Keras. "
                "Install tensorflow-cpu and h5py in the deployment environment."
            ) from keras_exc


def load_model_artifacts(model_dir: Optional[Path] = None):
    """Load the best available trained model artifacts for inference."""
    resolved_dir = Path(model_dir or MODEL_DIR).resolve()

    cnn_model_path = resolved_dir / "cnn_emotion_model.h5"
    cnn_encoder_path = resolved_dir / "cnn_label_encoder.pkl"
    cnn_config_path = resolved_dir / "cnn_config.json"
    legacy_model_path = resolved_dir / "emotion_model.pkl"
    scaler_path = resolved_dir / "scaler.pkl"
    encoder_path = resolved_dir / "label_encoder.pkl"

    if cnn_model_path.exists() and cnn_encoder_path.exists():
        try:
            model = _load_cnn_model(cnn_model_path)
        except ImportError:
            if legacy_model_path.exists() and scaler_path.exists() and encoder_path.exists():
                model = joblib.load(legacy_model_path)
                scaler = joblib.load(scaler_path)
                encoder = joblib.load(encoder_path)
                return {"model": model, "scaler": scaler, "encoder": encoder, "model_type": "mlp", "config": None}
            raise

        encoder = joblib.load(cnn_encoder_path)
        config = {}
        if cnn_config_path.exists():
            with cnn_config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
        return {"model": model, "scaler": None, "encoder": encoder, "model_type": "cnn", "config": config}

    if not legacy_model_path.exists() or not scaler_path.exists() or not encoder_path.exists():
        raise FileNotFoundError(
            f"Expected model files under {resolved_dir}, but one or more were not found."
        )

    model = joblib.load(legacy_model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)

    return {"model": model, "scaler": scaler, "encoder": encoder, "model_type": "mlp", "config": None}


def _load_audio_from_wav_bytes(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode PCM WAV bytes into a mono float32 numpy array."""
    wav_buffer = io.BytesIO(wav_bytes)

    with wave.open(wav_buffer, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        dtype = np.uint8
        scale = 128.0
    elif sample_width == 2:
        dtype = np.int16
        scale = 32768.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if sample_width == 1:
        audio = (audio - scale) / scale
    else:
        audio = audio / scale

    return audio, sample_rate


def _extract_lpc_features(audio: np.ndarray, order: int = 13) -> np.ndarray:
    """Approximate LPC coefficients for a hybrid MFCC+LPC feature vector."""
    if audio.size < 2:
        return np.zeros(order, dtype=np.float32)

    try:
        lpc_coeffs = librosa.lpc(audio, order=order)
    except Exception:
        lpc_coeffs = np.zeros(order, dtype=np.float32)

    if lpc_coeffs.size < order:
        padded = np.zeros(order, dtype=np.float32)
        padded[: lpc_coeffs.size] = lpc_coeffs.astype(np.float32)
        return padded

    return np.asarray(lpc_coeffs[:order], dtype=np.float32)


def extract_cnn_features_from_wav_bytes(
    wav_bytes: bytes,
    config: Optional[dict] = None,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Create CNN-compatible MFCC input tensors matching the training pipeline."""
    cfg = dict(DEFAULT_CNN_CONFIG)
    if config:
        cfg.update(config)

    audio, sample_rate = _load_audio_from_wav_bytes(wav_bytes)

    if sample_rate != target_sample_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)

    audio, _ = librosa.effects.trim(audio, top_db=20)

    if audio.size == 0:
        audio = np.zeros(target_sample_rate, dtype=np.float32)

    audio = librosa.util.normalize(audio)

    n_mfcc = int(cfg.get("n_mfcc", 40))
    max_pad_len = int(cfg.get("max_pad_len", 174))
    mfcc = librosa.feature.mfcc(y=audio, sr=target_sample_rate, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_pad_len]

    mfcc = mfcc[..., np.newaxis]
    return mfcc.astype(np.float32).reshape(1, n_mfcc, max_pad_len, 1)


def extract_features_from_wav_bytes(
    wav_bytes: bytes,
    expected_features: int = EXPECTED_FEATURES,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Create a hybrid MFCC+LPC feature vector consistent with the report."""
    audio, sample_rate = _load_audio_from_wav_bytes(wav_bytes)

    if sample_rate != target_sample_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)

    audio, _ = librosa.effects.trim(audio, top_db=20)

    if audio.size == 0:
        audio = np.zeros(target_sample_rate, dtype=np.float32)

    audio = librosa.util.normalize(audio)

    mfcc = librosa.feature.mfcc(y=audio, sr=target_sample_rate, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_std = np.std(mfcc.T, axis=0)
    lpc_features = _extract_lpc_features(audio, order=13)

    features = np.concatenate((mfcc_mean, mfcc_std, lpc_features)).astype(np.float32)
    features = features.reshape(1, -1)

    if features.shape[1] > expected_features:
        features = features[:, :expected_features]
    elif features.shape[1] < expected_features:
        pad_width = expected_features - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode="constant")

    return features
