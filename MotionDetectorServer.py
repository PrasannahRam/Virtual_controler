from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

# --------------------
# Load model ONCE
# --------------------
model = tf.keras.models.load_model("gesture_nn_best.h5")

# If you used string labels during training
LABELS = ["catch", "snap", "zoom"]  # adjust order if needed

app = Flask(__name__)

# --------------------
# Prediction endpoint
# --------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    if "features" not in data:
        return jsonify({"error": "No features provided"}), 400

    features = np.array(data["features"], dtype=np.float32)

    # Shape: (features,) → (1, features)
    features = np.expand_dims(features, axis=0)

    # Model prediction
    preds = model.predict(features, verbose=0)

    confidence = float(np.max(preds))
    label_index = int(np.argmax(preds))
    label = LABELS[label_index]

    # Unknown gesture rejection
    if confidence < 0.6:
        return jsonify({
            "gesture": "unknown",
            "confidence": confidence
        })

    return jsonify({
        "gesture": label,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
