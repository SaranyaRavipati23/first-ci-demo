"""
Practical 5: Student Result Prediction Flask API
Provides a REST API for predicting student pass/fail results
using the trained ML model.
"""

import os
import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_result_model.pkl")

# Feature columns — must match train_model.py FEATURE_COLS exactly
FEATURE_COLS = ["internal_marks", "attendance", "assignment_marks", "previous_score"]

app = Flask(__name__)

# Load the model once at startup
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def health():
    """Health-check endpoint. Returns service status."""
    return jsonify({"status": "ok", "service": "Student Result Prediction API"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict whether a student passes or fails.

    Expects a JSON body with the following numeric fields:
      - internal_marks   : marks in internal assessments
      - attendance       : attendance percentage
      - assignment_marks : marks in assignments
      - previous_score   : score from previous examination

    Returns:
      {
        "prediction": "PASS" | "FAIL",
        "prediction_code": 1 | 0
      }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    # Validate all required fields are present
    missing = [f for f in FEATURE_COLS if f not in data]
    if missing:
        return (
            jsonify({"error": "Missing required fields", "missing_fields": missing}),
            400,
        )

    # Validate all fields are numeric
    invalid = []
    for field in FEATURE_COLS:
        val = data[field]
        if not isinstance(val, (int, float)):
            invalid.append(field)
    if invalid:
        return (
            jsonify({"error": f"Fields must be numeric: {invalid}"}),
            400,
        )

    # Build input DataFrame in the correct column order
    input_df = pd.DataFrame(
        [[data[f] for f in FEATURE_COLS]],
        columns=FEATURE_COLS,
    )

    prediction_code = int(model.predict(input_df)[0])
    prediction_label = "PASS" if prediction_code == 1 else "FAIL"

    return jsonify(
        {
            "prediction": prediction_label,
            "prediction_code": prediction_code,
        }
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
