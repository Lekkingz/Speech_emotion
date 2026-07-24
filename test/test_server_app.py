import unittest

from server.app import create_app


class ServerAppTests(unittest.TestCase):
    def test_create_app_exposes_health_endpoint(self):
        app = create_app()
        client = app.test_client()

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_status_reports_cnn_model_when_artifacts_exist(self):
        app = create_app()
        client = app.test_client()

        response = client.get("/status")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["model_type"], "cnn")
        self.assertTrue(payload["model_loaded"])


if __name__ == "__main__":
    unittest.main()
