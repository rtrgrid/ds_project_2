import torch
import numpy as np
import sys, os
import plotly.express as px

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

X_train, y_train = create_windows(train_scaled, df.index[:len(train)], 24, 12)
X_test, y_test = create_windows(test_scaled, df.index[len(train):], 24, 12)

X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

# =========================
# LOAD MODEL
# =========================
model = TransformerModel(
    input_size=3,
    d_model=64,
    nhead=4,
    num_layers=2,
    horizon=12,
    positional_type="sinusoidal"
).to(DEVICE)

model.load_state_dict(torch.load("models/transformer_model.pth", map_location="cpu"))
model.eval()

# =========================
# GET PREDICTIONS
# =========================
with torch.no_grad():
    preds = model(X_test)

preds = preds.cpu().numpy()
y_true = y_test.cpu().numpy()

# =========================
# FIND BEST & WORST
# =========================
errors = np.mean(np.abs(preds - y_true), axis=1)

best_idx = np.argmin(errors)
worst_idx = np.argmax(errors)

print("Best sample:", best_idx, "Error:", errors[best_idx])
print("Worst sample:", worst_idx, "Error:", errors[worst_idx])

# =========================
# FUNCTION TO PLOT ATTENTION
# =========================
def plot_attention(sample_idx, title):
    x = X_test[sample_idx].unsqueeze(0)

    # forward pass
    _ = model(x)

    # get attention from layer 0
    attn_all = model.transformer[0].attn_weights.detach().numpy()

    for h in range(attn_all.shape[1]):
        attn = attn_all[0, h]

        fig = px.imshow(
            attn,
            labels=dict(x="Key", y="Query", color="Attention"),
            title=f"{title} - Head {h}"
        )

        # fig.show()
        file_name = f"plots/attention/{title.replace(' ', '_')}_head_{h}.html"
        fig.write_html(file_name)
        print("Saved:", file_name)

# =========================
# VISUALIZE
# =========================
plot_attention(best_idx, "Best Prediction")
plot_attention(worst_idx, "Worst Prediction")