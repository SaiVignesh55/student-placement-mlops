import unittest
from unittest.mock import patch
from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    @patch("app.load_model")
    def test_high_performance_prediction(self, mock_load_model):
        mock_model = mock_load_model.return_value
        mock_model.predict.return_value = [1]

        response = self.client.post(
            "/predict",
            json={
                "cgpa": 9.0,
                "internships": 2,
                "projects": 4,
                "aptitude_score": 90,
                "technical_skills_score": 90,
                "attendance": 95,
                "communication_score": 90
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["prediction"],
            "PLACED"
        )

    @patch("app.load_model")
    def test_low_performance_prediction(self, mock_load_model):
        mock_model = mock_load_model.return_value
        mock_model.predict.return_value = [0]

        response = self.client.post(
            "/predict",
            json={
                "cgpa": 5.5,
                "internships": 0,
                "projects": 0,
                "aptitude_score": 40,
                "technical_skills_score": 35,
                "attendance": 60,
                "communication_score": 40
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["prediction"],
            "NOT PLACED"
        )

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={
                "cgpa": 9.0,
                "internships": 2
            }
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "missing_fields",
            response.get_json()
        )


if __name__ == "__main__":
    unittest.main()
