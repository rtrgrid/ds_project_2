import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def run_arima(train, test, horizon=12):

    history = list(train)
    predictions = []

    for t in range(len(test) - horizon):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=horizon)
        predictions.append(forecast)
        history.append(test[t])

    return np.array(predictions)