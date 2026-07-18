import unittest

from server.app import create_app


class ServerAppTests(unittest.TestCase):
    def test_create_app_exposes_health_endpoint(self):
        app = create_app()
        client = app.test_client()

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
