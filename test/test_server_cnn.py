import io
import wave
import unittest

import numpy as np

from server.app import create_app


class DummyEncoder:
    def inverse_transform(self, values):
        return np.array(["happy" for _ in values])


class DummyModel:
    def predict(self, features, verbose=0):
        return np.array([[0.1, 0.9]])


class ServerCNNTests(unittest.TestCase):
    def test_cnn_predict_endpoint_works_without_scaler(self):
        app = create_app(model=DummyModel(), scaler=None, encoder=DummyEncoder(), model_type="cnn", config={})
        client = app.test_client()

        sample_rate = 22050
        duration_seconds = 0.2
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        pcm = (audio * 32767).astype(np.int16)

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())

        response = client.post(
            "/predict",
            data={"audio": (io.BytesIO(wav_buffer.getvalue()), "test.wav")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["emotion"], "happy")
        self.assertGreaterEqual(payload["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
