# SpeechEmotionSystem

This repository implements a real-time Speech Emotion Recognition system using a client-server architecture.

- ESP32: records audio from an INMP441 microphone, sends WAV data to the server, displays status on an SSD1306 display, and provides audio feedback through an I2S-compatible output stage.
- Python/Flask server: loads the trained model artifacts from `models/trained/` and exposes prediction endpoints via `/predict`.
- Signal processing: the inference pipeline follows the report’s hybrid approach by combining MFCC features with LPC-based coefficients before classification.

## Quick start

1. Create and activate a Python virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
.venv\Scripts\activate    # Windows PowerShell
```

2. Install the required dependencies.

```bash
pip install -r requirements.txt -r server/requirements.txt
```

3. Ensure the trained artifacts exist in `models/trained/`:

- `emotion_model.pkl`
- `scaler.pkl`
- `label_encoder.pkl`

4. Start the server.

```bash
python server/app.py
```

The server exposes:

- `GET /health` for a basic health check
- `GET /status` for model and feature metadata
- `POST /predict` for audio classification

## Validation

Run the unit tests with:

```bash
python -m unittest discover -s test -p "test_*.py" -v
```

## Hardware notes

The ESP32 firmware is organized as a PlatformIO project. Open the repository root in VS Code and build/upload the firmware from the PlatformIO extension.
