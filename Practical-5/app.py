import joblib
import pandas as pd
from flask import Flask, jsonify, request
from pathlib import Path

app = Flask(__name__)

MODEL_PATH = Path("student_placement_model.pkl")

FEATURES = [
    "cgpa",
    "internships",
    "projects",
    "aptitude_score",
    "technical_skills_score",
    "attendance",
    "communication_score"
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "student_placement_model.pkl was not found."
        )

    return joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "student-placement-prediction"
    })


@app.post("/predict")
def predict():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    sample = pd.DataFrame([{
        feature: data[feature]
        for feature in FEATURES
    }])

    model = load_model()

    prediction_code = int(model.predict(sample)[0])

    prediction = "PLACED" if prediction_code == 1 else "NOT PLACED"

    return jsonify({
        "prediction": prediction,
        "prediction_code": prediction_code
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
