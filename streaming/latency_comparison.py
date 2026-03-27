import time
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.transformer_model import TransformerModel
from src.data_loader import load_data
from src.preprocessing import scale_data

DEVICE = "cpu"

# ----------------------------
# LOAD MODEL
# ----------------------------
model = TransformerModel(
    input_size=3,
    d_model=128,
    nhead=4,
    num_layers=2,
    horizon=6
)

model.load_state_dict(torch.load("models/transformer_model.pth", map_location=DEVICE))
model.eval()

# ----------------------------
# LOAD DATA
# ----------------------------
df = load_data("data/Electric_Production.csv")
series = df["value"].values

scaled, _, scaler = scale_data(series, series)

context = 24

# ----------------------------
# FULL INFERENCE
# ----------------------------
start = time.time()

for i in range(context, len(scaled)):
    x = scaled[i-context:i]
    x = np.repeat(x.reshape(1, context, 1), 3, axis=2)
    x = torch.tensor(x, dtype=torch.float32)

    with torch.no_grad():
        _ = model(x)

end = time.time()

full_time = end - start

# ----------------------------
# STREAMING INFERENCE
# ----------------------------
window = scaled[:context].tolist()

start = time.time()

for i in range(context, len(scaled)):
    x = np.array(window[-context:])
    x = np.repeat(x.reshape(1, context, 1), 3, axis=2)
    x = torch.tensor(x, dtype=torch.float32)

    with torch.no_grad():
        _ = model(x)

    window.append(scaled[i])

end = time.time()

stream_time = end - start

# ----------------------------
# RESULTS
# ----------------------------
steps = len(scaled) - context

print("\n⚡ Latency Comparison\n")

print(f"Full Inference Time: {full_time:.4f}s")
print(f"Streaming Time: {stream_time:.4f}s")

print(f"\nPer Step Latency:")
print(f"Full: {full_time/steps*1000:.4f} ms")
print(f"Streaming: {stream_time/steps*1000:.4f} ms")