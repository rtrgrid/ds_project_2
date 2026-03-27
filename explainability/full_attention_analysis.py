import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from src.data_loader import load_data
from src.preprocessing import train_test_split, scale_data
from src.windows import create_windows
from src.transformer_model import TransformerModel

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =========================
# LOAD MODEL (MATCH TRAINED)
# =========================
model = TransformerModel(
    input_size=3,
    d_model=128,
    nhead=4,
    num_layers=2,
    horizon=6
).to(DEVICE)

model.load_state_dict(torch.load("models/transformer_model.pth", map_location=DEVICE))
model.eval()

print("✅ Model Loaded")

# =========================
# LOAD DATA
# =========================
df = load_data("data/Electric_Production.csv")
series = df["value"].values

train, test = train_test_split(series)
train_scaled, test_scaled, scaler = scale_data(train, test)

X_test, y_test = create_windows(test_scaled, df.index[len(train):], 24, 6)

X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

# =========================
# FORWARD PASS + ATTENTION
# =========================
with torch.no_grad():
    preds = model(X_test[:50])  # small batch

# collect attention
attn = model.get_attention_weights()  # assume list of layers

# shape: [layers, batch, heads, seq, seq]
attn = torch.stack(attn).cpu().numpy()

print("Attention shape:", attn.shape)

# =========================
# 1. HEATMAP (first sample)
# =========================
plt.imshow(attn[0,0,0], cmap="hot")
plt.title("Layer 0 Head 0 Attention")
plt.colorbar()
plt.savefig("outputs/attention_heatmap.png")
plt.close()

# =========================
# 2. AVERAGE ATTENTION DISTANCE
# =========================
seq_len = attn.shape[-1]
distances = np.arange(seq_len)

avg_dist = 0
count = 0

for l in range(attn.shape[0]):
    for h in range(attn.shape[2]):
        weights = attn[l, :, h]  # [batch, seq, seq]

        for b in range(weights.shape[0]):
            for t in range(seq_len):
                avg_dist += np.sum(weights[b, t] * distances)
                count += 1

avg_dist /= count

print("📊 Average Attention Distance:", avg_dist)

# =========================
# 3. IMPORTANT TIMESTEPS
# =========================
importance = np.mean(attn, axis=(0,1,2))  # [seq, seq]
importance = np.sum(importance, axis=0)

top_indices = np.argsort(importance)[-5:]

print("🔥 Most important timesteps:", top_indices)

# =========================
# 4. CORRECT vs WRONG
# =========================
preds = preds.cpu().numpy()
y_true = y_test[:50].cpu().numpy()

errors = np.mean((preds - y_true)**2, axis=1)

best_idx = np.argmin(errors)
worst_idx = np.argmax(errors)

print("Best sample:", best_idx)
print("Worst sample:", worst_idx)

# visualize comparison
plt.plot(attn[0, best_idx, 0], label="Best")
plt.plot(attn[0, worst_idx, 0], label="Worst")
plt.legend()
plt.title("Attention Comparison")
plt.savefig("outputs/attention_compare.png")
plt.close()

# =========================
# 5. SALIENCY MAP
# =========================
X_test.requires_grad = True

pred = model(X_test[0:1])
loss = pred.mean()
loss.backward()

saliency = X_test.grad[0].abs().cpu().numpy()

plt.plot(saliency[:,0])
plt.title("Saliency Map")
plt.savefig("outputs/saliency.png")
plt.close()

print("✅ Task 2 COMPLETED")