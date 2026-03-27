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
# TEST TYPES
# =========================
# types = ["sinusoidal", "learnable", "none"]
types = ["sinusoidal", "learnable", "none", "alibi"]

results = []

for t in types:
    print(f"\n🚀 Testing: {t}")

    model = TransformerModel(
        input_size=3,
        d_model=128,
        nhead=4,
        num_layers=2,
        horizon=6,
        positional_type=t
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = torch.nn.MSELoss()

    # quick training (10 epochs enough)
    for epoch in range(10):
        model.train()

        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()

    # evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test)

    preds = preds.cpu().numpy()
    y_true = y_test.cpu().numpy()

    rmse = np.sqrt(mean_squared_error(y_true.flatten(), preds.flatten()))

    print(f"{t} RMSE:", rmse)

    results.append((t, rmse))

# =========================
# FINAL RESULTS
# =========================
print("\n📊 FINAL RESULTS")
for r in results:
    print(r)