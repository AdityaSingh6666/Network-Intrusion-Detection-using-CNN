import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scripts.utils import load_nsl_kdd, preprocess_data

# ------------------------ Streamlit Page Config ------------------------
st.set_page_config(page_title="Network Intrusion Detection with CNN", layout="centered")
st.title("🔒 Network Intrusion Detection using CNN")
st.write("Upload your NSL-KDD CSV file to detect intrusions using your trained CNN model.")

# ------------------------ File Uploader ------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv", "txt"])

# ------------------------ Cached Model Loader ------------------------
@st.cache_resource
def load_cnn_model():
    return load_model("models/cnn_nids_model.keras")

if uploaded_file is not None:
    st.success("File uploaded successfully.")

    with st.spinner("Loading model..."):
        model = load_cnn_model()

    st.info("Preprocessing data...")

    temp_path = "temp_uploaded_file.csv"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    df = load_nsl_kdd(temp_path)
    X, _ = preprocess_data(df)

    # ------------------------ Reshape for CNN ------------------------
    st.write(f"X shape after preprocess_data: {X.shape}")

    if X.ndim == 2:
        if X.shape[1] < 144:
            pad_amt = 144 - X.shape[1]
            X = np.pad(X, ((0, 0), (0, pad_amt)), mode='constant')
        else:
            X = X[:, :144]
        X = X.reshape(-1, 12, 12, 1)

    st.success("Data preprocessed. Starting prediction...")

    # ------------------------ Model Prediction ------------------------
    predictions = model.predict(X, verbose=0)
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    if predictions.shape[1] == 1:
        predicted_classes = (predictions > 0.5).astype(int).reshape(-1)
    else:
        predicted_classes = np.argmax(predictions, axis=1)

    st.write(f"Predicted_classes shape: {predicted_classes.shape}")

    # ------------------------ Append Predictions ------------------------
    df['prediction'] = predicted_classes
    df['prediction_label'] = df['prediction'].map({0: 'Normal', 1: 'Attack'})

    normal_count = np.sum(predicted_classes == 0)
    attack_count = np.sum(predicted_classes == 1)

    # ------------------------ Display Results ------------------------
    st.subheader("📊 Prediction Summary")
    st.write(f"✅ **Normal connections:** {normal_count}")
    st.write(f"⚠️ **Attack connections:** {attack_count}")

    fig, ax = plt.subplots()
    ax.pie([normal_count, attack_count], labels=["Normal", "Attack"], autopct="%1.1f%%", startangle=90, colors=["#2ecc71", "#e74c3c"])
    ax.axis("equal")
    st.pyplot(fig)

    st.subheader("🔍 Sample Predictions")
    st.dataframe(df[['prediction', 'prediction_label']].value_counts().reset_index().rename(columns={0: 'count'}))

    # ------------------------ CSV Download ------------------------
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predictions CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv',
    )

    os.remove(temp_path)

else:
    st.info("Please upload a CSV file following NSL-KDD format for prediction.")
