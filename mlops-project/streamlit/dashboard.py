"""Streamlit dashboard for interactive scoring against the Production model."""

from __future__ import annotations

import os

import mlflow
import streamlit as st

from src.utils import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    get_sample_payload,
    infer_tracking_uri,
    load_environment,
    to_dataframe,
)

load_environment()
MODEL_NAME = os.getenv("MODEL_NAME", "customer-churn-classifier")

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("Customer Churn Inference Dashboard")
st.caption("Interactive scoring interface for the current MLflow Production model.")

mlflow.set_tracking_uri(infer_tracking_uri())
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
defaults = get_sample_payload()

with st.sidebar:
    st.header("Quick actions")
    if st.button("Load sample customer"):
        st.session_state["sample_loaded"] = True
    st.write("MLflow tracking URI:", infer_tracking_uri())

if st.session_state.get("sample_loaded"):
    defaults = get_sample_payload()

left, right = st.columns(2)
features: dict[str, object] = {}

with left:
    st.subheader("Numeric profile")
    for column in NUMERIC_FEATURES:
        features[column] = st.number_input(
            column.replace("_", " ").title(),
            value=float(defaults[column]),
        )

with right:
    st.subheader("Service profile")
    for column in BOOLEAN_FEATURES:
        features[column] = st.selectbox(
            column.replace("_", " ").title(),
            options=["yes", "no"],
            index=0 if defaults[column] == "yes" else 1,
        )
    for column in CATEGORICAL_FEATURES:
        options = {
            "contract_type": ["month-to-month", "one-year", "two-year"],
            "region": ["north", "south", "east", "west"],
            "internet_service": ["fiber", "dsl", "wireless"],
        }[column]
        features[column] = st.selectbox(
            column.replace("_", " ").title(),
            options=options,
            index=options.index(str(defaults[column])),
        )

if st.button("Predict churn risk", type="primary"):
    dataframe = to_dataframe(features)
    prediction = int(model.predict(dataframe)[0])
    probability = float(model.predict_proba(dataframe)[0][1])
    status = "High churn risk" if prediction == 1 else "Low churn risk"
    st.metric("Prediction", status)
    st.metric("Churn probability", f"{probability:.2%}")
