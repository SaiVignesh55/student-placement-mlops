def predict_placement(cgpa, internships, projects):
    if cgpa >= 7.0 and internships >= 1 and projects >= 2:
        return "PLACED"
    else:
        return "NOT PLACED"


if __name__ == "__main__":
    result = predict_placement(8.2, 2, 3)
    print("Placement Prediction:", result)
