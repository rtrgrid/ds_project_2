import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


def train_test_split(series, train_ratio=0.8):
    split_index = int(len(series) * train_ratio)
    train = series[:split_index]
    test = series[split_index:]
    return train, test


def scale_data(train, test, scaler_type="minmax"):

    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid scaler type")

    train = np.array(train).reshape(-1, 1)
    test = np.array(test).reshape(-1, 1)

    scaler.fit(train)

    train_scaled = scaler.transform(train).flatten()
    test_scaled = scaler.transform(test).flatten()

    return train_scaled, test_scaled, scaler