import io
import wave
from pathlib import Path
from typing import Optional, Tuple

import joblib
import librosa
import numpy as np


MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "trained"
TARGET_SAMPLE_RATE = 22050
EXPECTED_FEATURES = 80


def load_model_artifacts(model_dir: Optional[Path] = None):
    """Load the trained model artifacts once and return them."""
    resolved_dir = Path(model_dir or MODEL_DIR).resolve()

    model_path = resolved_dir / "emotion_model.pkl"
    scaler_path = resolved_dir / "scaler.pkl"
    encoder_path = resolved_dir / "label_encoder.pkl"

    if not model_path.exists() or not scaler_path.exists() or not encoder_path.exists():
        raise FileNotFoundError(
            f"Expected model files under {resolved_dir}, but one or more were not found."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)

    return model, scaler, encoder


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
