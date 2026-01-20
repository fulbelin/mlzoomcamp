import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

MODEL_PATH = Path("model.bin")

app = Flask("breast-cancer-risk")


def load_artifact():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


artifact = load_artifact()
pipeline = artifact["pipeline"]
feature_names = artifact["feature_names"]


def make_dataframe(payload: dict) -> pd.DataFrame:
    # allow partial input: missing features become NaN -> imputer handles them
    row = {k: payload.get(k, np.nan) for k in feature_names}
    return pd.DataFrame([row], columns=feature_names)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True) or {}
    X = make_dataframe(payload)
    proba = float(pipeline.predict_proba(X)[:, 1][0])

    pred = int(proba >= 0.5)
    result = {
        "malignancy_probability": proba,
        "prediction": "malignant" if pred == 1 else "benign",
        "model": artifact.get("model_name"),
    }
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # local dev
    app.run(host="0.0.0.0", port=9696, debug=True)
