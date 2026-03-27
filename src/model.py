import torch
from transformers import TimeSeriesTransformerConfig
from transformers import TimeSeriesTransformerForPrediction

def build_model(context=24, horizon=12, d_model=64):

    config = TimeSeriesTransformerConfig(
        context_length=context,
        prediction_length=horizon,
        input_size=1,              # 🔥 IMPORTANT
        d_model=d_model,
        lags_sequence=[1, 2],      # keep small lags
        num_time_features=1,
    )

    model = TimeSeriesTransformerForPrediction(config)
    return model

