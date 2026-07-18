import io
import wave
import unittest

import numpy as np

from python_ai.inference.model_utils import extract_features_from_wav_bytes


class FeatureExtractionTests(unittest.TestCase):
    def test_extract_features_returns_expected_shape(self):
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

        features = extract_features_from_wav_bytes(wav_buffer.getvalue(), expected_features=80)

        self.assertEqual(features.shape, (1, 80))


if __name__ == "__main__":
    unittest.main()
