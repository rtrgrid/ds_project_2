import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import time
import coremltools as ct
import numpy as np

from src.transformer_model import TransformerModel

# config
context = 24
input_size = 3
horizon = 6

# load pytorch model
model = TransformerModel(input_size, 128, 4, 2, horizon)
model.load_state_dict(torch.load("models/transformer_model.pth", map_location="cpu"))
model.eval()

# load coreml
mlmodel = ct.models.MLModel("models/transformer.mlpackage")

def benchmark(batch_size):
    x = torch.randn(batch_size, 24, 3)  # ✅ FIXED shape

    print("Shape:", x.shape)

    # PyTorch
    start = time.time()
    with torch.no_grad():
        _ = model(x)
    pt_time = time.time() - start

    # CoreML
    x_np = x.numpy()
    start = time.time()
    _ = mlmodel.predict({"x_1": x_np})
    cm_time = time.time() - start

    return pt_time, cm_time

results = []

for b in [1, 32, 64]:
    print(f"\nBatch {b}")

    x = torch.randn(b, 24, 3)

    # PyTorch works for all batch sizes
    start = time.time()
    with torch.no_grad():
        _ = model(x)
    pt_time = time.time() - start

    # CoreML only for batch=1
    if b == 1:
        x_np = x.numpy()
        start = time.time()
        _ = mlmodel.predict({"x_1": x_np})
        cm_time = time.time() - start
    else:
        cm_time = None

    results.append((b, pt_time, cm_time))

    print(f"PyTorch: {pt_time:.4f}s")
    if cm_time:
        print(f"CoreML: {cm_time:.4f}s")
    else:
        print("CoreML: Not supported (fixed batch size)")


# FINAL TABLE
print("\n📊 FINAL LATENCY TABLE")
for b, pt, cm in results:
    if cm:
        print(f"Batch {b} → PyTorch: {pt:.4f}s | CoreML: {cm:.4f}s")
    else:
        print(f"Batch {b} → PyTorch: {pt:.4f}s | CoreML: Not supported")