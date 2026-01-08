from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model

model = load_model("gesture_nn_best.h5")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

app = Flask(__name__)

# --------------------
# Prediction endpoint
# --------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Expecting sequence: [[dx,dy,dz], ...]
    sequence = data["sequence"]

    features = np.array(sequence).flatten()
    features_scaled = scaler.transform([features])

    prediction = model.predict(features_scaled)
    class_id = np.argmax(prediction)
    label = le.inverse_transform([class_id])[0]



    return jsonify({
        "prediction": label,
        "confidence": float(np.max(prediction))
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
