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

# =========================
# TRAIN CONTEXT (24)
# =========================
context_train = 24
horizon = 12

# =========================
# TEST CONTEXTS (LONGER)
# =========================
contexts_test = [48, 96]

# =========================
# POSITIONAL TYPES
# =========================
types = ["sinusoidal", "learnable", "none", "alibi"]

results = []

for t in types:
    print(f"\n🚀 Testing: {t}")

    # train model
    model = TransformerModel(
        input_size=3,
        d_model=64,
        nhead=4,
        num_layers=2,
        horizon=horizon,
        positional_type=t
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = torch.nn.MSELoss()

    X_train, y_train = create_windows(train_scaled, df.index[:len(train)], context_train, horizon)

    

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    # quick training
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()

    # =========================
    # TEST ON LONGER CONTEXT
    # =========================
    for ctx in contexts_test:

        X_test, y_test = create_windows(test_scaled, df.index[len(train):], ctx, horizon)

        if len(X_test) == 0:
            print(f"{t} → Context {ctx} skipped (not enough data)")
            continue

        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            preds = model(X_test)

        rmse = np.sqrt(mean_squared_error(
            y_test.numpy().flatten(),
            preds.numpy().flatten()
        ))

        print(f"{t} → Context {ctx} RMSE: {rmse:.4f}")

        results.append((t, ctx, rmse))

# =========================
# FINAL RESULTS
# =========================
print("\n📊 EXTRAPOLATION RESULTS")
for r in results:
    print(r)