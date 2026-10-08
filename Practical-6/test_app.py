import unittest
from app import app


class TestPlacementAPI(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_home(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "cgpa": 8.5,
                "internships": 2,
                "projects": 3,
                "aptitude_score": 85
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.get_json())


if __name__ == "__main__":
    unittest.main()
