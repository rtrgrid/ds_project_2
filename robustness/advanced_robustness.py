import torch
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_data
from src.preprocessing import train_test_split, scale_data
from src.windows import create_windows
from src.transformer_model import TransformerModel

from sklearn.metrics import mean_squared_error

DEVICE = torch.device("cpu")

# =========================
# LOAD DATA
# =========================
df = load_data("data/Electric_Production.csv")
series = df["value"].values

train, test = train_test_split(series)
train_scaled, test_scaled, scaler = scale_data(train, test)

X_train, y_train = create_windows(train_scaled, df.index[:len(train)], 24, 12)
X_test, y_test = create_windows(test_scaled, df.index[len(train):], 24, 12)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# =========================
# MODEL
# =========================
model = TransformerModel(3, 64, 4, 2, 12, positional_type="sinusoidal")
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fn = torch.nn.MSELoss()

# =========================
# ADVERSARIAL TRAINING
# =========================
epsilon = 0.05

print("\n🚀 Adversarial Training")

for epoch in range(10):
    model.train()

    X_train.requires_grad = True

    pred = model(X_train)
    loss = loss_fn(pred, y_train)
    loss.backward()

    # FGSM perturbation
    X_adv = X_train + epsilon * torch.sign(X_train.grad)

    optimizer.zero_grad()

    # train on both clean + adversarial
    pred_clean = model(X_train.detach())
    pred_adv = model(X_adv.detach())

    loss = loss_fn(pred_clean, y_train) + loss_fn(pred_adv, y_train)
    loss.backward()
    optimizer.step()

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(x, y):
    with torch.no_grad():
        preds = model(x)
    return np.sqrt(mean_squared_error(y.numpy().flatten(), preds.numpy().flatten()))

# =========================
# CLEAN PERFORMANCE
# =========================
clean_rmse = evaluate(X_test, y_test)

# =========================
# NOISE TESTS
# =========================

# 1. Gaussian noise
noise = torch.randn_like(X_test) * 0.05
X_noise = X_test + noise

noise_rmse = evaluate(X_noise, y_test)

# 2. Missing values
X_missing = X_test.clone()
X_missing[:, :5, :] = 0

missing_rmse = evaluate(X_missing, y_test)

# 3. FGSM attack
X_test.requires_grad = True

pred = model(X_test)
loss = loss_fn(pred, y_test)
loss.backward()

X_fgsm = X_test + epsilon * torch.sign(X_test.grad)
fgsm_rmse = evaluate(X_fgsm.detach(), y_test)

# =========================
# RESULTS
# =========================
print("\n📊 ROBUSTNESS RESULTS")
print(f"Clean RMSE: {clean_rmse:.4f}")
print(f"Noise RMSE: {noise_rmse:.4f}")
print(f"Missing RMSE: {missing_rmse:.4f}")
print(f"FGSM RMSE: {fgsm_rmse:.4f}")

# =========================
# VULNERABILITY SCORE
# =========================
vulnerability = fgsm_rmse - clean_rmse

print(f"\n⚠️ Vulnerability Score: {vulnerability:.4f}")