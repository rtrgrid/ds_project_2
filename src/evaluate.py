import numpy as np
import torch
from sklearn.metrics import mean_squared_error

def evaluate_model(model, X_test, y_test, scaler, device):

    model.eval()
    
    batch_size = X_test.shape[0]

    past_time_features = torch.zeros(
        (batch_size, X_test.shape[1], 1),
        dtype=torch.float32
    ).to(device)

    past_observed_mask = torch.ones_like(X_test).to(device)

    with torch.no_grad():
        outputs = model(
            past_values=X_test,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
        )

        preds = outputs.prediction_outputs


def naive_baseline(test_series, context=24, horizon=12):
    naive_preds = []
    for i in range(len(test_series) - context - horizon):
        last_value = test_series[i+context-1]
        naive_preds.append([last_value] * horizon)
    return np.array(naive_preds)
