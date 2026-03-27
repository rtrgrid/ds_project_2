import torch
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.transformer_model import TransformerModel

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = TransformerModel(
    input_size=3,
    d_model=128,
    nhead=4,
    num_layers=2,
    horizon=6
).to(DEVICE)

model.load_state_dict(torch.load("models/transformer_model.pth", map_location=DEVICE))
model.eval()

x = torch.randn(1, 24, 3).to(DEVICE)

# =========================
# FULL INFERENCE
# =========================
start = time.time()

for _ in range(500):
    with torch.no_grad():
        model(x)

end = time.time()

full_time = (end - start)/500 * 1000

# =========================
# PARTIAL CONTEXT
# =========================
start = time.time()

for _ in range(500):
    with torch.no_grad():
        model(x[:, -12:, :])

end = time.time()

partial_time = (end - start)/500 * 1000

print("\n⚡ FINAL LATENCY RESULTS")
print(f"Full Context: {full_time:.4f} ms")
print(f"Partial Context: {partial_time:.4f} ms")