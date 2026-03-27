import time
import torch
import numpy as np
import onnxruntime as ort
import pandas as pd
import sys
import os

# ----------------------------
# Fix import path
# ----------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.transformer_model import TransformerModel

# ----------------------------
# Config
# ----------------------------
BATCH_SIZES = [1, 32, 64]
SEQ_LEN = 24
FEATURES = 3
DEVICE = "cpu"

# ----------------------------
# Load PyTorch Model
# ----------------------------
print("Loading PyTorch model...")

model = TransformerModel(
    input_size=3,
    d_model=64,
    nhead=4,
    num_layers=2,
    horizon=12
)

state_dict = torch.load("models/transformer_model.pth", map_location=DEVICE)
model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

pt_model = model

print("PyTorch expects input features:", model.input_projection.in_features)

# ----------------------------
# Load ONNX Model
# ----------------------------
print("Loading ONNX model...")

onnx_session = ort.InferenceSession("models/transformer.onnx")

onnx_input = onnx_session.get_inputs()[0]
onnx_input_name = onnx_input.name

print("ONNX input name:", onnx_input_name)
print("ONNX expected shape:", onnx_input.shape)

# ----------------------------
# Warmup
# ----------------------------
def warmup():
    # PyTorch can handle batch
    x_pt = torch.randn(32, SEQ_LEN, FEATURES).to(DEVICE)

    # ONNX ONLY supports batch=1 (static)
    x_onnx = np.random.randn(1, SEQ_LEN, FEATURES).astype(np.float32)

    for _ in range(10):
        with torch.no_grad():
            _ = pt_model(x_pt)

        _ = onnx_session.run(None, {onnx_input_name: x_onnx})

# ----------------------------
# Benchmark Functions
# ----------------------------
def benchmark_pytorch(batch_size):
    x = torch.randn(batch_size, SEQ_LEN, FEATURES).to(DEVICE)

    start = time.time()
    with torch.no_grad():
        _ = pt_model(x)
    end = time.time()

    return (end - start) * 1000


def benchmark_onnx(batch_size):
    x = np.random.randn(1, SEQ_LEN, FEATURES).astype(np.float32)

    start = time.time()

    # simulate batching
    for _ in range(batch_size):
        _ = onnx_session.run(None, {onnx_input_name: x})

    end = time.time()

    return (end - start) * 1000


# ----------------------------
# Run Benchmark
# ----------------------------
print("\nRunning benchmarks...\n")

warmup()

results = []

for batch in BATCH_SIZES:
    pt_times = []
    onnx_times = []

    for _ in range(20):
        pt_times.append(benchmark_pytorch(batch))
        onnx_times.append(benchmark_onnx(batch))

    pt_avg = np.mean(pt_times)
    onnx_avg = np.mean(onnx_times)
    speedup = pt_avg / onnx_avg

    results.append({
        "batch_size": batch,
        "pytorch_ms": pt_avg,
        "onnx_ms": onnx_avg,
        "speedup": speedup
    })

    print(f"Batch {batch}: PyTorch={pt_avg:.2f} ms | ONNX={onnx_avg:.2f} ms | Speedup={speedup:.2f}x")

# ----------------------------
# Save Results
# ----------------------------
df = pd.DataFrame(results)
df.to_csv("deployment/latency_results.csv", index=False)

print("\n✅ Saved → deployment/latency_results.csv")