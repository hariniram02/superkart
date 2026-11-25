import os
import joblib
import pandas as pd
from flask import Flask, jsonify, request

superkart_api = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "superkart_model.joblib")
model = joblib.load(MODEL_PATH)

KEY_ALIASES = {
    "Product_Id_char": "Product_Category_Prefix",
    "Store_Age_Years": "Store_Age",
    "Product_Type_Category": "Product_Type",
}


def normalize_record(record):
    row = {KEY_ALIASES.get(k, k): v for k, v in record.items()}
    if "Store_Establishment_Year" not in row and "Store_Age" in row:
        row["Store_Establishment_Year"] = 2026 - int(row["Store_Age"])
    return row


def predict_rows(payload):
    records = payload if isinstance(payload, list) else [payload]
    if not records:
        raise ValueError("Payload must not be empty")
    normalized = [normalize_record(r) for r in records]
    df_input = pd.DataFrame(normalized)
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
