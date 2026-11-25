import os
import joblib
import pandas as pd
from flask import Flask, jsonify, request

superkart_api = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "superkart_model.joblib")

model = joblib.load(MODEL_PATH)


def predict_rows(payload):
    records = payload if isinstance(payload, list) else [payload]
    if not records:
        raise ValueError("Payload must not be empty")
    df_input = pd.DataFrame(records)
    preds = model.predict(df_input)
    return [float(p) for p in preds]


@superkart_api.post("/v1/predict")
def predict_single():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON"}), 400
    try:
        preds = predict_rows(data)
        return jsonify({"predictions": preds})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@superkart_api.post("/v1/predictbatch")
def predict_batch():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON"}), 400
    try:
        preds = predict_rows(data)
        return jsonify({"predictions": preds, "count": len(preds)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=5000)
