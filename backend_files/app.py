import io
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


def predict_from_dataframe(df_input):
    if df_input.empty:
        raise ValueError("Input dataset is empty")
    records = [normalize_record(r) for r in df_input.to_dict(orient="records")]
    normalized_df = pd.DataFrame(records)
    preds = model.predict(normalized_df)
    return [float(p) for p in preds]


def predict_rows(payload):
    records = payload if isinstance(payload, list) else [payload]
    if not records:
        raise ValueError("Payload must not be empty")
    return predict_from_dataframe(pd.DataFrame(records))


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
    try:
        if "file" in request.files:
            uploaded_file = request.files["file"]
            raw_bytes = uploaded_file.read()
            df_batch = pd.read_csv(io.BytesIO(raw_bytes))
            preds = predict_from_dataframe(df_batch)
            return jsonify({"predictions": preds, "count": len(preds)})

        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Expected multipart CSV file 'file' or JSON payload"}), 400
        preds = predict_rows(data)
        return jsonify({"predictions": preds, "count": len(preds)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=5000)
