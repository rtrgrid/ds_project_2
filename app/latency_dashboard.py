import streamlit as st
import torch
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.transformer_model import TransformerModel

# ---------------------------
# CONFIG
# ---------------------------
context = 24
input_size = 3
horizon = 6

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    model = TransformerModel(input_size, 128, 4, 2, horizon)
    model.load_state_dict(torch.load("models/transformer_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ---------------------------
# UI CONFIG
# ---------------------------
st.set_page_config(page_title="⚡ Electricity Forecasting", layout="wide")

st.title("⚡ Electricity Forecasting Dashboard")

# ---------------------------
# TABS
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📊 Latency", "🔮 Predict", "📂 Upload Data"])

# ===========================
# TAB 1: LATENCY
# ===========================
with tab1:
    st.subheader("Model Latency Comparison")

    data = pd.DataFrame({
        "Batch": [1, 32, 64],
        "PyTorch": [0.0010, 0.0036, 0.0048],
        "CoreML": [0.0183, None, None]
    })

    st.dataframe(data, use_container_width=True)
    st.line_chart(data.set_index("Batch"))

    st.info("CoreML optimized for real-time edge inference (batch=1)")
    st.warning("PyTorch scales better for batch processing")

# ===========================
# TAB 2: LIVE PREDICTION
# ===========================
with tab2:
    st.subheader("🔮 Live Forecast")

    st.write("Enter last 24 timesteps (3 features each)")

    # user input
    user_input = np.random.rand(1, context, input_size)

    if st.button("Run Prediction"):
        x = torch.tensor(user_input, dtype=torch.float32)

        with torch.no_grad():
            pred = model(x).numpy()

        st.success("Prediction generated!")

        st.write("Output:")
        st.write(pred)

        st.line_chart(pred.flatten())

# ===========================
# TAB 3: CSV UPLOAD
# ===========================
with tab3:
    st.subheader("📂 Upload Data for Prediction")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("Preview:")
        st.dataframe(df.head())

        # ✅ FIX: take ONLY numeric column
        values = df.select_dtypes(include=[np.number]).values.flatten()

        if len(values) < 24:
            st.error("Need at least 24 rows")
        else:
            if st.button("Predict from CSV"):

                # last 24 timesteps
                data = values[-24:]

                # normalize (important!)
                data = (data - np.mean(data)) / (np.std(data) + 1e-8)

                # make 3 features
                data = np.stack([data, data, data], axis=1)

                data = np.expand_dims(data, axis=0).astype(np.float32)

                x = torch.tensor(data)

                with torch.no_grad():
                    pred = model(x).numpy()

                st.success("Prediction Complete!")

                st.write("Predictions:")
                st.write(pred)

                st.line_chart(pred.flatten())
# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.caption("🚀 Transformer-based Electricity Forecasting | Edge AI Ready")