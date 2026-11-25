import os
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "superkart_model.joblib")

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "service": "superkart-backend"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Invalid or missing JSON body"}), 400

        records = payload if isinstance(payload, list) else [payload]
        if len(records) == 0:
            return jsonify({"error": "Input list is empty"}), 400

        input_df = pd.DataFrame(records)
        preds = model.predict(input_df)

        return jsonify({
            "predictions": [float(p) for p in preds],
            "count": len(preds),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
