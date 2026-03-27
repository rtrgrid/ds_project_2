import torch
import numpy as np
import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.transformer_model import TransformerModel
from src.data_loader import load_data
from src.preprocessing import train_test_split, scale_data

# =========================
# CONFIG
# =========================
context = 24
input_size = 3
horizon = 6

DEVICE = torch.device("cpu")

# =========================
# LOAD MODEL
# =========================
model = TransformerModel(input_size, 128, 4, 2, horizon).to(DEVICE)
model.load_state_dict(torch.load("models/transformer_model.pth", map_location=DEVICE))
model.eval()

print("🔥 Using NEW Custom Transformer Model")
print("🚀 Streaming with Circular Buffer")

# =========================
# LOAD DATA
# =========================
df = load_data("data/Electric_Production.csv")
series = df["value"].values

train, test = train_test_split(series)
train_scaled, test_scaled, scaler = scale_data(train, test)

# =========================
# INIT BUFFER
# =========================
buffer = list(train_scaled[-context:])  # last 24 points

predictions = []
errors = []

# =========================
# STREAMING LOOP
# =========================
start_time = time.time()

for i in range(len(test_scaled)):

    # prepare input
    data = np.array(buffer[-context:])

    # make 3 features
    data = np.stack([data, data, data], axis=1)
    data = np.expand_dims(data, axis=0).astype(np.float32)

    x = torch.tensor(data).to(DEVICE)

    # inference
    with torch.no_grad():
        pred = model(x)

    pred_value = pred[0, 0].item()  # first horizon step

    predictions.append(pred_value)

    # compute error (drift check)
    true_value = test_scaled[i]
    error = abs(pred_value - true_value)
    errors.append(error)

    # update buffer (circular)
    buffer.append(true_value)
    if len(buffer) > context:
        buffer.pop(0)

    # logging
    if i % 50 == 0:
        print(f"Step {i}: Prediction={pred_value:.2f}, Error={error:.4f}")

# =========================
# METRICS
# =========================
end_time = time.time()
total_time = end_time - start_time

latency = total_time / len(predictions)
throughput = len(predictions) / total_time

avg_error = np.mean(errors)

# =========================
# RESULTS
# =========================
print("\n✅ Streaming complete")
print(f"Total predictions: {len(predictions)}")
print(f"Total Time: {total_time:.4f}s")
print(f"Per Step Latency: {latency * 1000:.4f} ms")
print(f"Throughput: {throughput:.2f} predictions/sec")
print(f"Average Error (Drift): {avg_error:.4f}")