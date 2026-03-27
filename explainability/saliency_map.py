import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.transformer_model import TransformerModel
from src.data_loader import load_data
from src.preprocessing import train_test_split, scale_data
from src.windows import create_windows

# =========================
# CONFIG
# =========================
context = 24
input_size = 3
horizon = 6

# =========================
# LOAD MODEL
# =========================
model = TransformerModel(input_size, 128, 4, 2, horizon)
model.load_state_dict(torch.load("models/transformer_model.pth", map_location="cpu"))
model.eval()

print("✅ Model Loaded")

# =========================
# LOAD DATA
# =========================
df = load_data("data/Electric_Production.csv")
series = df["value"].values

train, test = train_test_split(series)
train_scaled, test_scaled, scaler = scale_data(train, test)

X_test, y_test = create_windows(test_scaled, df.index[len(train):], context, horizon)

X_test = torch.tensor(X_test, dtype=torch.float32)

# =========================
# PICK SAMPLE
# =========================
x = X_test[0:1].clone().detach()
x.requires_grad = True

# =========================
# FORWARD + BACKWARD
# =========================
output = model(x)

# use sum to get scalar
loss = output.sum()

loss.backward()

# =========================
# SALIENCY
# =========================
saliency = x.grad.abs().mean(dim=-1).squeeze().detach().numpy()  # (24,)

print("Saliency shape:", saliency.shape)

# =========================
# PLOT
# =========================
plt.figure(figsize=(8, 4))
plt.plot(saliency, marker='o')
plt.title("Saliency Map (Input Importance)")
plt.xlabel("Timestep")
plt.ylabel("Importance")
plt.grid()

plt.show()