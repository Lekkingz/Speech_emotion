import sys
from pathlib import Path

from flask import Flask, jsonify, request
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from python_ai.inference.model_utils import (
    EXPECTED_FEATURES,
    extract_cnn_features_from_wav_bytes,
    extract_features_from_wav_bytes,
    load_model_artifacts,
)


def create_app(model=None, scaler=None, encoder=None, model_type=None, config=None):
    app = Flask(__name__)

    if model is None or encoder is None:
        try:
            artifacts = load_model_artifacts()
            model = artifacts["model"]
            scaler = artifacts["scaler"]
            encoder = artifacts["encoder"]
            model_type = artifacts["model_type"]
            config = artifacts.get("config")
        except FileNotFoundError as exc:
            model, scaler, encoder = None, None, None
            model_type = "unknown"
            config = None
            app.config["MODEL_ERROR"] = str(exc)
        else:
            app.config["MODEL_ERROR"] = None
    else:
        app.config["MODEL_ERROR"] = None

    app.config["MODEL"] = model
    app.config["SCALER"] = scaler
    app.config["ENCODER"] = encoder
    app.config["MODEL_TYPE"] = model_type or "mlp"
    app.config["CNN_CONFIG"] = config or {}

    @app.route("/health", methods=["GET"])
    def health():
        model_loaded = model is not None and scaler is not None and encoder is not None
        return jsonify({
            "status": "ok",
            "service": "speech-emotion-server",
            "model_loaded": model_loaded,
        }), 200

    @app.route("/status", methods=["GET"])
    def status():
        encoder_labels = []
        if encoder is not None:
            encoder_labels = encoder.classes_.tolist()

        return jsonify({
            "status": "ready" if encoder is not None else "degraded",
            "model_loaded": encoder is not None,
            "model_type": app.config.get("MODEL_TYPE", "mlp"),
            "expected_features": EXPECTED_FEATURES,
            "labels": encoder_labels,
            "model_error": app.config.get("MODEL_ERROR"),
        })

    @app.route("/predict", methods=["POST"])
    def predict():
        if model is None or scaler is None or encoder is None:
            return jsonify({"error": "Model artifacts are not loaded"}), 503

        payload = None
        if "audio" in request.files:
            payload = request.files["audio"].read()
        else:
            payload = request.get_data(cache=False)

        if not payload:
            return jsonify({"error": "No audio payload received"}), 400

        try:
            model_type = app.config.get("MODEL_TYPE", "mlp")
            if model_type == "cnn":
                features = extract_cnn_features_from_wav_bytes(payload, config=app.config.get("CNN_CONFIG", {}))
                probabilities = model.predict(features, verbose=0)[0]
                class_idx = int(np.argmax(probabilities))
                confidence = float(probabilities[class_idx])
            else:
                features = extract_features_from_wav_bytes(payload, expected_features=EXPECTED_FEATURES)
                transformed = scaler.transform(features)

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(transformed)[0]
                    class_idx = int(np.argmax(probabilities))
                    confidence = float(probabilities[class_idx])
                else:
                    prediction = model.predict(transformed)[0]
                    class_idx = int(prediction)
                    confidence = 1.0

            emotion = encoder.inverse_transform([class_idx])[0]
            return jsonify({
                "emotion": str(emotion),
                "confidence": round(confidence, 4),
            })
        except Exception as exc:  # pragma: no cover - defensive guard
            return jsonify({"error": str(exc)}), 500

    @app.route("/", methods=["GET"])
    def root():
        return jsonify({"message": "Speech Emotion Recognition Server"})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
