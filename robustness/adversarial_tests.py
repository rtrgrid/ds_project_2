import torch
import numpy as np
import sys
import os

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

X_train, y_train = create_windows(train_scaled, df.index[:len(train)], 24, 6)
X_test, y_test = create_windows(test_scaled, df.index[len(train):], 24, 6)

X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

# =========================
# MODEL
# =========================
model = TransformerModel(
    input_size=3,
    d_model=128,
    nhead=4,
    num_layers=2,
    horizon=6
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fn = torch.nn.MSELoss()

# quick training
for epoch in range(10):
    model.train()
    optimizer.zero_grad()

    pred = model(X_train)
    loss = loss_fn(pred, y_train)

    loss.backward()
    optimizer.step()

print("✅ Model trained")

# =========================
# FGSM ATTACK
# =========================
def fgsm_attack(model, x, y, epsilon):
    x_adv = x.clone().detach().requires_grad_(True)

    pred = model(x_adv)
    loss = loss_fn(pred, y)

    loss.backward()

    x_adv = x_adv + epsilon * x_adv.grad.sign()
    return x_adv.detach()


# =========================
# PGD ATTACK
# =========================
def pgd_attack(model, x, y, epsilon, alpha=0.01, steps=5):
    x_adv = x.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)

        pred = model(x_adv)
        loss = loss_fn(pred, y)

        loss.backward()

        x_adv = x_adv + alpha * x_adv.grad.sign()

        # clip within epsilon ball
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)

        x_adv = x_adv.detach()

    return x_adv


# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(x, y):
    model.eval()
    with torch.no_grad():
        pred = model(x)

    pred = pred.cpu().numpy()
    y_true = y.cpu().numpy()

    return np.sqrt(mean_squared_error(y_true.flatten(), pred.flatten()))


# =========================
# TEST EPSILONS
# =========================
epsilons = [0, 0.01, 0.05, 0.1]

print("\n📊 FGSM Results")
for eps in epsilons:
    if eps == 0:
        rmse = evaluate(X_test, y_test)
    else:
        x_adv = fgsm_attack(model, X_test, y_test, eps)
        rmse = evaluate(x_adv, y_test)

    print(f"Epsilon={eps} → RMSE={rmse:.4f}")


print("\n📊 PGD Results")
for eps in epsilons:
    if eps == 0:
        rmse = evaluate(X_test, y_test)
    else:
        x_adv = pgd_attack(model, X_test, y_test, eps)
        rmse = evaluate(x_adv, y_test)

    print(f"Epsilon={eps} → RMSE={rmse:.4f}")

def fgsm_attack(x, epsilon, grad):
    return x + epsilon * grad.sign()


def adversarial_training(model, X, y, epochs=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for _ in range(epochs):
        for i in range(len(X)):
            x = X[i:i+1].clone().detach().requires_grad_(True)
            target = y[i:i+1]

            output = model(x)
            loss = loss_fn(output, target)
            loss.backward()

            # create adversarial
            adv_x = fgsm_attack(x, 0.05, x.grad)

            # mix normal + adversarial
            optimizer.zero_grad()
            out_clean = model(x.detach())
            out_adv = model(adv_x.detach())

            loss = loss_fn(out_clean, target) + loss_fn(out_adv, target)
            loss.backward()
            optimizer.step()

    return model

print("\n🚀 Running Adversarial Training...")

model_adv = adversarial_training(model, X_train, y_train)

# evaluate new model
model_adv.eval()
with torch.no_grad():
    preds_adv = model_adv(X_test)

rmse_adv = np.sqrt(mean_squared_error(
    y_test.cpu().numpy().flatten(),
    preds_adv.cpu().numpy().flatten()
))

print(f"Adversarial Trained RMSE: {rmse_adv}")