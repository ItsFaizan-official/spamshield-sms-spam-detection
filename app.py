"""Streamlit interface for SpamShield."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.database import get_prediction_history, init_db, save_prediction

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "spam_classifier.joblib"
METADATA_PATH = PROJECT_DIR / "models" / "model_metadata.json"

st.set_page_config(
    page_title="SpamShield | SMS Spam Detection",
    page_icon="🛡️",
    layout="centered",
)
st.title("🛡️ SpamShield")
st.subheader("Intelligent SMS Spam Detection")
st.write(
    "Check a text message with an NLP model trained to distinguish "
    "unwanted spam from legitimate ham."
)

init_db(PROJECT_DIR / "spamshield.db")

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.warning(
        "The trained model is not available yet. Run "
        "`python train.py` in the project folder, then refresh this page."
    )
    st.stop()


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


def clear_message() -> None:
    st.session_state["message_input"] = ""


model, metadata = load_artifacts()
model_name = metadata.get("model_name", "Trained classifier")
st.info(f"Active model: **{model_name}**")

message = st.text_area(
    "Enter an SMS message",
    key="message_input",
    placeholder="Example: Your delivery is on its way...",
    height=120,
    max_chars=5000,
)

predict_column, clear_column = st.columns(2)
predict_clicked = predict_column.button(
    "Predict message",
    type="primary",
    width="stretch",
)
clear_column.button(
    "Clear input",
    on_click=clear_message,
    width="stretch",
)

if predict_clicked:
    cleaned_message = message.strip()
    if not cleaned_message:
        st.error("Please enter a message before classifying it.")
    else:
        prediction = str(model.predict([cleaned_message])[0])
        save_prediction(
            cleaned_message,
            prediction,
            model_name,
            PROJECT_DIR / "spamshield.db",
        )
        if prediction == "spam":
            st.error("🚨 SPAM — treat this message with caution.")
        else:
            st.success("✅ HAM — the model considers this message legitimate.")
        st.caption(
            "Predictions can be wrong. Do not rely on this result alone for "
            "financial or safety decisions."
        )

st.divider()
st.subheader("Recent prediction history")
history = get_prediction_history(
    limit=10,
    db_path=PROJECT_DIR / "spamshield.db",
)
if history:
    history_frame = pd.DataFrame(history)
    history_frame["original_message"] = history_frame["original_message"].str.slice(0, 100)
    st.dataframe(history_frame, width="stretch", hide_index=True)
else:
    st.write("No predictions have been stored yet. Try classifying a message above.")

with st.expander("Held-out test evaluation"):
    metrics = metadata.get("test_metrics", {})
    if metrics:
        cols = st.columns(4)
        for column, key, label in zip(
            cols,
            ["accuracy", "spam_precision", "spam_recall", "spam_f1"],
            ["Accuracy", "Spam precision", "Spam recall", "Spam F1"],
        ):
            column.metric(label, f"{metrics[key]:.3f}")
    st.caption("Measured on the untouched test split during training.")
