import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.transformer_model import TransformerModel

DEVICE = "cpu"

# ----------------------------
# Load Model (same config!)
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
# Run Model
# ----------------------------
x = torch.randn(1, 24, 3)

with torch.no_grad():
    _ = model(x)

# ----------------------------
# Collect Attention
# ----------------------------
attentions = []

for layer in model.transformer:
    attn = layer.attn_weights.detach().numpy()  # (1, 4, 24, 24)
    attentions.append(attn)

# ----------------------------
# Plot Heatmaps
# ----------------------------
os.makedirs("plots/attention", exist_ok=True)

for layer_idx, attn in enumerate(attentions):

    # remove batch dim → (4, 24, 24)
    attn = attn[0]

    for head_idx in range(attn.shape[0]):

        plt.figure(figsize=(6, 5))

        plt.imshow(attn[head_idx], aspect='auto')
        plt.colorbar()

        plt.title(f"Layer {layer_idx} - Head {head_idx}")
        plt.xlabel("Key (Past timestep)")
        plt.ylabel("Query (Current timestep)")

        plt.savefig(f"plots/attention/layer{layer_idx}_head{head_idx}.png")
        plt.close()

print("✅ Attention heatmaps saved in plots/attention/")