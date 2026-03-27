import itertools
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.data_loader import load_data
from src.preprocessing import train_test_split, scale_data
from src.windows import create_windows
from src.arima_model import run_arima
from src.transformer_model import TransformerModel
from src.lstm_model import LSTMModel


def run_transformer(train, test, context, horizon, d_model, scaler_type, epochs=50):

    train_scaled, test_scaled, scaler = scale_data(train, test, scaler_type)

    train_dates = pd.date_range(start="2000-01", periods=len(train), freq="M")
    test_dates = pd.date_range(start="2000-01", periods=len(test), freq="M")

    X_train, y_train = create_windows(train_scaled, train_dates, context, horizon)
    X_test, y_test = create_windows(test_scaled, test_dates, context, horizon)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    model = TransformerModel(
        input_size=3,
        d_model=d_model,
        nhead=4,
        num_layers=2,
        horizon=horizon
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = torch.nn.MSELoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train)
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test)

    preds = preds.cpu().numpy()
    y_true = y_test.cpu().numpy()

    preds = scaler.inverse_transform(preds.reshape(-1,1)).reshape(preds.shape)
    y_true = scaler.inverse_transform(y_true.reshape(-1,1)).reshape(y_true.shape)

    rmse = np.sqrt(mean_squared_error(y_true.flatten(), preds.flatten()))
    return rmse


def main():

    df = load_data("data/Electric_Production.csv")
    series = df["value"].values
    train, test = train_test_split(series)

    scalers = ["standard", "minmax", "robust"]
    contexts = [24, 48, 96]
    horizons = [6, 12, 24]
    d_models = [32, 64, 128]

    results = []

    for scaler, context, horizon, d_model in itertools.product(
        scalers, contexts, horizons, d_models
    ):
        try:
            rmse = run_transformer(
                train, test,
                context=context,
                horizon=horizon,
                d_model=d_model,
                scaler_type=scaler
            )

            print(f"{scaler} | C{context} | H{horizon} | D{d_model} → {rmse:.3f}")

            results.append({
                "Scaler": scaler,
                "Context": context,
                "Horizon": horizon,
                "D_model": d_model,
                "RMSE": rmse
            })

        except:
            continue

    results_df = pd.DataFrame(results)
    results_df.sort_values("RMSE", inplace=True)
    results_df.to_csv("experiment_results.csv", index=False)

    print("\nTop 5 Results:")
    print(results_df.head())


if __name__ == "__main__":
    main()