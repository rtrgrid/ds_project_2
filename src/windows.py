import numpy as np

def create_windows(series, dates, context=48, horizon=12):
    X, y = [], []

    for i in range(len(series) - context - horizon):
        past_values = series[i:i+context]
        future_values = series[i+context:i+context+horizon]

        past_dates = dates[i:i+context]

        # Create seasonal features
        months = np.array([d.month for d in past_dates])
        month_sin = np.sin(2 * np.pi * months / 12)
        month_cos = np.cos(2 * np.pi * months / 12)

        # Stack features
        features = np.stack([past_values, month_sin, month_cos], axis=1)

        X.append(features)
        y.append(future_values)

    return np.array(X), np.array(y)