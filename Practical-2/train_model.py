import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_dataset():
    rng = np.random.default_rng(42)
    number_of_students = 300

    data = pd.DataFrame({
        "cgpa": rng.uniform(5.5, 9.5, number_of_students),
        "internships": rng.integers(0, 4, number_of_students),
        "projects": rng.integers(0, 6, number_of_students),
        "aptitude_score": rng.integers(40, 101, number_of_students),
        "technical_skills_score": rng.integers(40, 101, number_of_students),
        "attendance": rng.integers(50, 101, number_of_students),
        "communication_score": rng.integers(40, 101, number_of_students)
    })

    weighted_score = (
        0.25 * data["cgpa"] * 10
        + 0.15 * data["internships"] * 25
        + 0.10 * data["projects"] * 20
        + 0.15 * data["aptitude_score"]
        + 0.15 * data["technical_skills_score"]
        + 0.10 * data["attendance"]
        + 0.10 * data["communication_score"]
    )

    # Add small random noise so the model is not unrealistically perfect
    noise = rng.normal(0, 5, number_of_students)
    weighted_score = weighted_score + noise

    # 1 = PLACED, 0 = NOT PLACED
    data["placed"] = (weighted_score >= 65).astype(int)

    return data


def train_model():
    print("Creating Student Placement dataset...")

    data = create_dataset()

    data.to_csv("student_placement.csv", index=False)

    print("Dataset created successfully.")
    print("Number of records:", len(data))

    features = [
        "cgpa",
        "internships",
        "projects",
        "aptitude_score",
        "technical_skills_score",
        "attendance",
        "communication_score"
    ]

    X = data[features]
    y = data["placed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    print("Training Student Placement model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))

    print("\nConfusion Matrix:")
    print(matrix)

    joblib.dump(model, "student_placement_model.pkl")

    print("\nModel saved as student_placement_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test)
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved as metrics.json")

    return accuracy


if __name__ == "__main__":
    train_model()
