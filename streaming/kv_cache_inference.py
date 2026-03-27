import torch
import numpy as np
import time
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

scaled, _, _ = scale_data(series, series)

context = 24

# ----------------------------
# NORMAL STREAMING
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

normal_time = end - start

# ----------------------------
# "CACHED" (SIMULATED)
# ----------------------------
# reuse tensor and shift instead of recreate

x = np.repeat(np.array(window[:context]).reshape(1, context, 1), 3, axis=2)
x = torch.tensor(x, dtype=torch.float32)

start = time.time()

for i in range(context, len(scaled)):
    # shift left
    x[:, :-1, :] = x[:, 1:, :].clone()

    # insert new value
    new_val = scaled[i]
    # new_val = np.repeat([[new_val]], 3, axis=1)
    # x[:, -1, :] = torch.tensor(new_val, dtype=torch.float32)
    new_val = torch.tensor([[new_val]*3], dtype=torch.float32)
    x[:, -1, :] = new_val

    with torch.no_grad():
        _ = model(x)

end = time.time()

cached_time = end - start

# ----------------------------
# RESULTS
# ----------------------------
steps = len(scaled) - context

print("\n🚀 KV Cache Simulation\n")

print(f"Normal Streaming Time: {normal_time:.4f}s")
print(f"Cached Streaming Time: {cached_time:.4f}s")

print(f"\nPer Step:")
print(f"Normal: {normal_time/steps*1000:.4f} ms")
print(f"Cached: {cached_time/steps*1000:.4f} ms")