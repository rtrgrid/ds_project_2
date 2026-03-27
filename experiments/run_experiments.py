import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from src.data_loader import load_data
from src.preprocessing import train_test_split, scale_data
from src.windows import create_windows
from src.transformer_model import TransformerModel


# =========================
# CREATE OUTPUT FOLDERS
# =========================
os.makedirs("outputs/predictions", exist_ok=True)
os.makedirs("models", exist_ok=True)  # 🔥 ensure models folder exists

# =========================
# EXPERIMENT SETTINGS
# =========================
contexts = [12, 24, 36, 48, 60, 72]
horizons = [6, 12, 18, 24]
d_models = [32, 64, 128]
epochs_list = [30, 40, 50, 60, 70, 80, 90, 100]
batch_size = 32

results = []

# 🔥 Track best model
best_rmse = float("inf")

# =========================
# LOAD DATA
# =========================
df = load_data("data/Electric_Production.csv")
series = df["value"].values

train, test = train_test_split(series)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

exp_id = 0

# =========================
# EXPERIMENT LOOP
# =========================
for context in contexts:
    for horizon in horizons:

        print(f"\nPreparing windows for Context={context} Horizon={horizon}")

        train_scaled, test_scaled, scaler = scale_data(train, test)

        train_dates = df.index[:len(train)]
        test_dates = df.index[len(train):]

        X_train, y_train = create_windows(train_scaled, train_dates, context, horizon)
        X_test, y_test = create_windows(test_scaled, test_dates, context, horizon)

        if len(X_train) == 0 or len(X_test) == 0:
            print("Skipping due to insufficient windows")
            continue

        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for d_model in d_models:

            print(f"Running Experiment {exp_id}")
            print(f"Context={context} Horizon={horizon} d_model={d_model}")

            # =========================
            # MODEL
            # =========================
            model = TransformerModel(
                input_size=X_train.shape[-1],
                d_model=d_model,
                nhead=4,
                num_layers=2,
                horizon=horizon
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
            loss_fn = torch.nn.MSELoss()

            # =========================
            # TRAINING
            # =========================
            for epoch in epochs_list:

                model.train()
                epoch_loss = 0

                for xb, yb in train_loader:
                    optimizer.zero_grad()

                    preds = model(xb)
                    loss = loss_fn(preds, yb)

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1} Loss {epoch_loss/len(train_loader):.4f}")

            # =========================
            # PREDICTION
            # =========================
            model.eval()

            with torch.no_grad():
                preds = model(X_test)

            preds = preds.cpu().numpy()
            y_true = y_test.cpu().numpy()

            preds = scaler.inverse_transform(
                preds.reshape(-1, 1)
            ).reshape(preds.shape)

            y_true = scaler.inverse_transform(
                y_true.reshape(-1, 1)
            ).reshape(y_true.shape)

            rmse = np.sqrt(
                mean_squared_error(y_true.flatten(), preds.flatten())
            )

            print("RMSE:", rmse)

            # =========================
            # 🔥 SAVE BEST MODEL
            # =========================
            if rmse < best_rmse:
                best_rmse = rmse
                torch.save(model.state_dict(), "models/transformer_model.pth")
                print("🔥 Best model saved! RMSE:", rmse)

            # =========================
            # SAVE FORECAST PLOT
            # =========================
            plt.figure(figsize=(8,4))

            plt.plot(y_true[0], label="Actual")
            plt.plot(preds[0], label="Prediction")

            plt.title(f"Context={context} | Horizon={horizon} | d_model={d_model}")
            plt.legend()
            plt.grid()

            plt.savefig(f"outputs/predictions/plot_{exp_id}.png")
            plt.close()

            # =========================
            # SAVE PREDICTIONS
            # =========================
            np.save(
                f"outputs/predictions/config_{exp_id}.npy",
                preds
            )

            # =========================
            # SAVE RESULTS
            # =========================
            results.append({
                "experiment_id": exp_id,
                "context": context,
                "horizon": horizon,
                "d_model": d_model,
                "rmse": rmse
            })

            exp_id += 1

# =========================
# SAVE RESULTS TABLE
# =========================
results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/experiment_results.csv",
    index=False
)

print("\n===============================")
print("Experiments completed.")
print("Total experiments:", len(results_df))
print("Best RMSE:", best_rmse)
print("=================================")