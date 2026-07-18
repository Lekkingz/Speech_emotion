import argparse
import io
import os
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from flask import Flask, jsonify, request

try:
    from .model_utils import (
        EXPECTED_FEATURES,
        load_model_artifacts,
        extract_features_from_wav_bytes,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from model_utils import (
        EXPECTED_FEATURES,
        load_model_artifacts,
        extract_features_from_wav_bytes,
    )


app = Flask(__name__)

MODEL, SCALER, ENCODER = load_model_artifacts()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "speech-emotion-server"})


@app.route("/status", methods=["GET"])
def status():
    return jsonify(
        {
            "status": "ready",
            "model_loaded": True,
            "expected_features": EXPECTED_FEATURES,
            "labels": ENCODER.classes_.tolist(),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_data(cache=False)

    if not payload:
        return jsonify({"error": "No audio payload received"}), 400

    try:
        features = extract_features_from_wav_bytes(payload, expected_features=EXPECTED_FEATURES)
        features = SCALER.transform(features)

        prediction = MODEL.predict(features)[0]
        probabilities = MODEL.predict_proba(features)[0]
        emotion = ENCODER.inverse_transform([prediction])[0]
        confidence = float(np.max(probabilities))

        return jsonify({"emotion": emotion, "confidence": round(confidence, 4)})
    except Exception as exc:  # pragma: no cover - defensive guard
        return jsonify({"error": str(exc)}), 500


@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Speech Emotion Recognition Server"})


def record_audio_seconds(duration_seconds: float, sample_rate: int = 22050) -> bytes:
    """Record PCM audio from the local microphone and return it as WAV bytes."""
    frames = int(duration_seconds * sample_rate)
    recording = sd.rec(int(frames), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()

    audio = np.clip(recording[:, 0], -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())

    return wav_buffer.getvalue()


def run_local_demo(duration_seconds: float = 4.0) -> None:
    print("Recording audio for prediction...")
    wav_bytes = record_audio_seconds(duration_seconds)

    features = extract_features_from_wav_bytes(wav_bytes, expected_features=EXPECTED_FEATURES)
    features = SCALER.transform(features)

    prediction = MODEL.predict(features)[0]
    probabilities = MODEL.predict_proba(features)[0]
    emotion = ENCODER.inverse_transform([prediction])[0]
    confidence = float(np.max(probabilities))

    print(f"Detected emotion: {emotion} ({confidence:.2%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition Flask server")
    parser.add_argument("--local", action="store_true", help="Run a local microphone demo instead of the server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if args.local:
        run_local_demo()
    else:
        app.run(host=args.host, port=args.port, debug=False)
