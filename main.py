import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from src.data_loader import load_data
from src.preprocessing import train_test_split, scale_data
from src.windows import create_windows
from src.arima_model import run_arima
from src.lstm_model import LSTMModel
from src.transformer_model import TransformerModel

import os

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


def main():

    # =========================
    # SETTINGS (SPEC MATCHED)
    # =========================
    context = 24
    horizon = 12
    epochs = 50
    batch_size = 32

    print("========== ELECTRICITY FORECAST EXPERIMENT ==========")
    print(f"Context Length : {context}")
    print(f"Prediction Len : {horizon}")
    print(f"Epochs         : {epochs}")
    print(f"Batch Size     : {batch_size}")
    print("=====================================================\n")

    # =========================
    # LOAD DATA
    # =========================
    df = load_data("data/Electric_Production.csv")
    series = df["value"].values
    train, test = train_test_split(series)

    # =========================
    # NAIVE BASELINE
    # =========================
    print("Running Naive Baseline...")

    naive_preds, y_true_naive = [], []

    for i in range(len(test) - horizon):
        naive_preds.append(np.repeat(test[i], horizon))
        y_true_naive.append(test[i:i+horizon])

    naive_preds = np.array(naive_preds)
    y_true_naive = np.array(y_true_naive)

    naive_rmse = np.sqrt(
        mean_squared_error(y_true_naive.flatten(), naive_preds.flatten())
    )

    print("Naive RMSE:", naive_rmse)
    np.save("outputs/naive_preds.npy", naive_preds)
    np.save("outputs/y_true_naive.npy", y_true_naive)

    print("Naive predictions saved to outputs/")

    # =========================
    # ARIMA
    # =========================
    print("\nRunning ARIMA...")

    arima_preds = run_arima(train, test, horizon=horizon)

    y_true_arima = np.array([
        test[i:i+horizon] for i in range(len(test) - horizon)
    ])

    arima_rmse = np.sqrt(
        mean_squared_error(y_true_arima.flatten(), arima_preds.flatten())
    )

    print("ARIMA RMSE:", arima_rmse)
    np.save("outputs/arima_preds.npy", arima_preds)
    np.save("outputs/y_true_arima.npy", y_true_arima)

    print("ARIMA predictions saved to outputs/")

    # =========================
    # SCALE DATA
    # =========================
    train_scaled, test_scaled, scaler = scale_data(train, test)

    train_dates = df.index[:len(train)]
    test_dates = df.index[len(train):]

    X_train, y_train = create_windows(train_scaled, train_dates, context, horizon)
    X_test, y_test = create_windows(test_scaled, test_dates, context, horizon)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Safety squeeze
    if X_train.dim() == 4:
        X_train = X_train.squeeze(-1)
        X_test = X_test.squeeze(-1)

    print("Final X_train shape:", X_train.shape)

    # =========================
    # CREATE DATALOADER
    # =========================
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.MSELoss()

    # =========================
    # LSTM
    # =========================
    print("\nRunning LSTM...")

    lstm_model = LSTMModel(
        input_size=X_train.shape[-1],
        hidden_size=64,
        horizon=horizon
    ).to(device)

    optimizer_lstm = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        lstm_model.train()
        epoch_loss = 0

        for xb, yb in train_loader:
            optimizer_lstm.zero_grad()
            preds = lstm_model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer_lstm.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"LSTM Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

    lstm_model.eval()
    with torch.no_grad():
        lstm_preds = lstm_model(X_test)

    lstm_preds = lstm_preds.cpu().numpy()
    y_true = y_test.cpu().numpy()

    lstm_preds = scaler.inverse_transform(
        lstm_preds.reshape(-1,1)
    ).reshape(lstm_preds.shape)

    y_true = scaler.inverse_transform(
        y_true.reshape(-1,1)
    ).reshape(y_true.shape)

    lstm_rmse = np.sqrt(
        mean_squared_error(y_true.flatten(), lstm_preds.flatten())
    )

    print("LSTM RMSE:", lstm_rmse)

    torch.save(lstm_model.state_dict(), "models/lstm_model.pth")
    print("LSTM model saved to models/lstm_model.pth")
    # =========================
    # TRANSFORMER
    # =========================
    print("\nRunning Transformer...")

    transformer_model = TransformerModel(
        input_size=X_train.shape[-1],
        d_model=64,
        nhead=4,
        num_layers=2,
        horizon=horizon
    ).to(device)

    optimizer_trans = torch.optim.Adam(transformer_model.parameters(), lr=5e-4)

    for epoch in range(epochs):
        transformer_model.train()
        epoch_loss = 0

        for xb, yb in train_loader:
            optimizer_trans.zero_grad()
            preds = transformer_model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer_trans.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Transformer Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

    transformer_model.eval()
    with torch.no_grad():
        transformer_preds = transformer_model(X_test)

    transformer_preds = transformer_preds.cpu().numpy()

    transformer_preds = scaler.inverse_transform(
        transformer_preds.reshape(-1,1)
    ).reshape(transformer_preds.shape)

    transformer_rmse = np.sqrt(
        mean_squared_error(y_true.flatten(), transformer_preds.flatten())
    )

    print("Transformer RMSE:", transformer_rmse)
    torch.save(transformer_model.state_dict(), "models/transformer_model.pth")
    print("Transformer model saved to models/transformer_model.pth")

    config = {
        "context_length": context,
        "horizon": horizon,
        "epochs": epochs,
        "batch_size": batch_size
    }

    pd.DataFrame([config]).to_csv("outputs/training_config.csv", index=False)

    print("Training config saved to outputs/training_config.csv")
    # =========================
    # FINAL RESULTS
    # =========================
    print("\n========== FINAL RESULTS ==========")
    print(f"Naive RMSE        : {naive_rmse:.4f}")
    print(f"ARIMA RMSE        : {arima_rmse:.4f}")
    print(f"LSTM RMSE         : {lstm_rmse:.4f}")
    print(f"Transformer RMSE  : {transformer_rmse:.4f}")
    print("===================================")

    

    results = {
        "Model": ["Naive", "ARIMA", "LSTM", "Transformer"],
        "RMSE": [naive_rmse, arima_rmse, lstm_rmse, transformer_rmse]
    }

    results_df = pd.DataFrame(results)

    results_df.to_csv("outputs/model_results.csv", index=False)

    print("Results saved to outputs/model_results.csv")

    models = ["Naive", "ARIMA", "LSTM", "Transformer"]
    rmse_values = [naive_rmse, arima_rmse, lstm_rmse, transformer_rmse]

    plt.figure(figsize=(8,5))
    bars = plt.bar(models, rmse_values)

    bars[np.argmin(rmse_values)].set_color("green")

    plt.title("Model RMSE Comparison")
    plt.ylabel("RMSE")
    plt.grid(axis="y")

    plt.savefig("plots/rmse_comparison.png")

    print("Plot saved to plots/rmse_comparison.png")

    # import torch

    # torch.save(lstm_model.state_dict(), "models/lstm_model.pth")
    # torch.save(transformer_model.state_dict(), "models/transformer_model.pth")

if __name__ == "__main__":
    main()