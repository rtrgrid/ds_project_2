import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction

from src.data_loader import load_data
from src.preprocessing import train_test_split, scale_data


print("Loading dataset...")

# =========================
# Load Data
# =========================
df = load_data("data/Electric_Production.csv")
series = df["value"].values

train, test = train_test_split(series)

print(f"Total rows: {len(series)}")
print(f"Train size: {len(train)}")
print(f"Test size: {len(test)}")

# =========================
# Parameters
# =========================
context = 72
horizon = 12
max_lag = 1   # because we will use lags_sequence=[1]

print(f"\nContext Length: {context}")
print(f"Prediction Length: {horizon}")

# =========================
# Scale Data
# =========================
train_scaled, test_scaled, scaler = scale_data(train, test, "standard")

# =========================
# Transformer Config
# =========================
print("\nInitializing Transformer...")

config = TimeSeriesTransformerConfig(
    prediction_length=horizon,
    context_length=context,
    lags_sequence=[1],     # minimal lag (cannot be empty)
    num_time_features=1,
    num_static_categorical_features=0,
    num_static_real_features=0,
    input_size=1,
    d_model=32,
)

model = TimeSeriesTransformerForPrediction(config)

# =========================
# Prepare Correct Inputs
# =========================
try:
    # IMPORTANT:
    # Must provide context + max_lag history
    history_length = context + max_lag

    past_values = torch.tensor(
        train_scaled[-history_length:], dtype=torch.float32
    ).unsqueeze(0).unsqueeze(-1)   # [1, context+lag, 1]

    # time features only for context_length
    past_time_features = torch.zeros(
        (1, context, 1), dtype=torch.float32
    )

    past_observed_mask = torch.ones(
        (1, context, 1), dtype=torch.float32
    )

    future_time_features = torch.zeros(
        (1, horizon, 1), dtype=torch.float32
    )

    print("\nInput Shapes:")
    print("past_values:", past_values.shape)
    print("past_time_features:", past_time_features.shape)
    print("past_observed_mask:", past_observed_mask.shape)
    print("future_time_features:", future_time_features.shape)

    print("\nRunning forward pass...")

    outputs = model(
        past_values=past_values,
        past_time_features=past_time_features,
        past_observed_mask=past_observed_mask,
        future_time_features=future_time_features
    )

    print("\n✅ Transformer ran successfully.")
    print("Output shape:", outputs.prediction_outputs.shape)

except Exception as e:
    print("\n❌ TRANSFORMER FAILED")
    print("Error Type:", type(e).__name__)
    print("Error Message:")
    print(e)