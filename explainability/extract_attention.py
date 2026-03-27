import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.transformer_model import TransformerModel

DEVICE = "cpu"

model = TransformerModel(
    input_size=3,
    d_model=128,   # ✅ FIXED
    nhead=4,
    num_layers=2,
    horizon=6      # ✅ FIXED
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
# Extract Attention
# ----------------------------
attentions = []

for layer in model.transformer:
    attn = layer.attn_weights  # stored during forward
    attentions.append(attn.detach().numpy())

print("Layers:", len(attentions))
print("Shape:", attentions[0].shape)