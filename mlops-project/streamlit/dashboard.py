"""Simple Streamlit UI for production model inference."""

from __future__ import annotations

import os

import mlflow
import streamlit as st

from src.utils import get_feature_template, infer_tracking_uri, to_dataframe

MODEL_NAME = os.getenv("MODEL_NAME", "breast-cancer-classifier")

st.set_page_config(page_title="MLOps Dashboard", layout="wide")
st.title("Production Model Inference Dashboard")

mlflow.set_tracking_uri(infer_tracking_uri())
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")

features = {}
st.write("Input all features and click Predict.")
for column in get_feature_template():
    features[column] = st.number_input(column, value=0.0, format="%.6f")

if st.button("Predict"):
    df = to_dataframe(features)
    pred = int(model.predict(df)[0])
    st.success(f"Prediction: {pred} (1=malignant, 0=benign)")
