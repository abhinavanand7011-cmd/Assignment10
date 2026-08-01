"""
app.py

Heart Disease Prediction - Flask REST API
===========================================
Covers Task 3 (API Development).

Loads the trained model (model.pkl) and exposes a /predict endpoint that
accepts patient details as JSON and returns a prediction as JSON.

Run locally:
    python app.py
Then POST to http://localhost:5000/predict
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model, scaler, and feature metadata once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
scaler = artifact["scaler"]
feature_names = artifact["feature_names"]
model_accuracy = artifact["accuracy"]


@app.route("/", methods=["GET"])
def home():
    """Simple landing page confirming the API is running."""
    return jsonify({
        "message": "Heart Disease Prediction API is running.",
        "usage": "POST patient details as JSON to /predict",
        "required_fields": feature_names,
        "model_test_accuracy": round(model_accuracy, 4)
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint (useful for Render / uptime monitors)."""
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept patient details as JSON and return a prediction as JSON.

    Expected JSON body, e.g.:
    {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }
    """
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    # Validate all required fields are present
    missing_fields = [f for f in feature_names if f not in data]
    if missing_fields:
        return jsonify({
            "error": "Missing required fields.",
            "missing_fields": missing_fields,
            "required_fields": feature_names
        }), 400

    try:
        # Build the feature vector in the exact order the model was trained on
        input_values = [[float(data[f]) for f in feature_names]]
    except (ValueError, TypeError):
        return jsonify({"error": "All fields must be numeric."}), 400

    input_scaled = scaler.transform(input_values)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"

    return jsonify({
        "prediction": result,
        "prediction_class": int(prediction),
        "probability_of_heart_disease": round(float(probability), 4)
    })


if __name__ == "__main__":
    # Render sets the PORT environment variable; default to 5000 for local runs
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
