from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "Student Placement Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    cgpa = data.get("cgpa", 0)
    internships = data.get("internships", 0)
    projects = data.get("projects", 0)
    aptitude_score = data.get("aptitude_score", 0)

    score = (
        cgpa * 10
        + internships * 5
        + projects * 5
        + aptitude_score
    ) / 2

    prediction = "Placed" if score >= 50 else "Not Placed"

    return jsonify({
        "prediction": prediction
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
