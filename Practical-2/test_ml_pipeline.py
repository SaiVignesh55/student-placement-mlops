import unittest
import json
from pathlib import Path

import joblib
import pandas as pd

from train_model import train_model


class TestMLPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Train the model once before running the tests
        train_model()

    def test_dataset_created(self):
        self.assertTrue(Path("student_placement.csv").exists())

    def test_model_created(self):
        self.assertTrue(Path("student_placement_model.pkl").exists())

    def test_metrics_created(self):
        self.assertTrue(Path("metrics.json").exists())

    def test_dataset_has_expected_columns(self):
        data = pd.read_csv("student_placement.csv")

        expected_columns = [
            "cgpa",
            "internships",
            "projects",
            "aptitude_score",
            "technical_skills_score",
            "attendance",
            "communication_score",
            "placed"
        ]

        for column in expected_columns:
            self.assertIn(column, data.columns)

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]

        self.assertGreaterEqual(accuracy, 0.50)
        self.assertLessEqual(accuracy, 1.00)

    def test_model_can_predict(self):
        model = joblib.load("student_placement_model.pkl")

        sample = pd.DataFrame([{
            "cgpa": 8.5,
            "internships": 2,
            "projects": 3,
            "aptitude_score": 85,
            "technical_skills_score": 88,
            "attendance": 90,
            "communication_score": 85
        }])

        prediction = model.predict(sample)[0]

        self.assertIn(prediction, [0, 1])

    def test_high_performance_student(self):
        model = joblib.load("student_placement_model.pkl")

        sample = pd.DataFrame([{
            "cgpa": 9.0,
            "internships": 3,
            "projects": 5,
            "aptitude_score": 95,
            "technical_skills_score": 95,
            "attendance": 95,
            "communication_score": 95
        }])

        prediction = model.predict(sample)[0]

        self.assertEqual(prediction, 1)

    def test_low_performance_student(self):
        model = joblib.load("student_placement_model.pkl")

        sample = pd.DataFrame([{
            "cgpa": 5.5,
            "internships": 0,
            "projects": 0,
            "aptitude_score": 40,
            "technical_skills_score": 40,
            "attendance": 50,
            "communication_score": 40
        }])

        prediction = model.predict(sample)[0]

        self.assertEqual(prediction, 0)


if __name__ == "__main__":
    unittest.main()
