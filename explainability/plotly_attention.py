import torch
import numpy as np
import plotly.express as px
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.transformer_model import TransformerModel

# config
model = TransformerModel(3, 64, 4, 2, 12, positional_type="sinusoidal")
model.load_state_dict(torch.load("models/transformer_model.pth", map_location="cpu"))
model.eval()

# dummy input
x = torch.randn(1, 24, 3)

# forward pass
_ = model(x)

# get attention (layer 0)
attn = model.transformer[0].attn_weights.detach().numpy()[0, 0]

# plot
fig = px.imshow(attn,
    labels=dict(x="Key", y="Query", color="Attention"),
    title="Attention Heatmap (Layer 0 Head 0)"
)

fig.write_html("plots/attention/plotly_heatmap.html")
fig.show()