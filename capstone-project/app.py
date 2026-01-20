import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path("model.bin")

st.set_page_config(page_title="Breast Cancer Risk", layout="centered")
st.title("Breast Cancer Risk Stratification")
st.write("Predict **malignant vs benign** from input features. Demo project (not medical advice).")

@st.cache_resource
def load_artifact():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

artifact = load_artifact()
pipeline = artifact["pipeline"]
feature_names = artifact["feature_names"]

st.subheader("Inputs")

# Provide a friendly subset first; missing values will be imputed by the pipeline
default_fields = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
]

inputs = {}
for f in default_fields:
    inputs[f] = st.number_input(f, value=float("nan"))

with st.expander("Advanced: set additional features"):
    for f in feature_names:
        if f in default_fields:
            continue
        inputs[f] = st.number_input(f, value=float("nan"))

def make_df(payload: dict) -> pd.DataFrame:
    row = {k: payload.get(k, np.nan) for k in feature_names}
    return pd.DataFrame([row], columns=feature_names)

if st.button("Predict"):
    X = make_df(inputs)
    proba = float(pipeline.predict_proba(X)[:, 1][0])
    pred = "malignant" if proba >= 0.5 else "benign"

    st.metric("Malignancy probability", f"{proba:.3f}")
    st.write(f"Prediction: **{pred}**")
