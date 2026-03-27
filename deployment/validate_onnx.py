import torch
import onnxruntime as ort
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.transformer_model import TransformerModel

# config
context = 24
input_size = 3
horizon = 6

# load pytorch
model = TransformerModel(input_size, 128, 4, 2, horizon)
model.load_state_dict(torch.load("models/transformer_model.pth", map_location="cpu"))
model.eval()

# ONNX session
session = ort.InferenceSession("models/transformer.onnx")

# test input
x = torch.randn(1, context, input_size)

# pytorch output
with torch.no_grad():
    pt_out = model(x).numpy()

# onnx output
onnx_out = session.run(None, {"input": x.numpy()})[0]

# compare
diff = np.max(np.abs(pt_out - onnx_out))

print("Max Difference:", diff)

if diff < 1e-5:
    print("✅ ONNX VALIDATION PASSED")
else:
    print("❌ ONNX VALIDATION FAILED")