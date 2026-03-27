import torch
import coremltools as ct
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.transformer_model import TransformerModel

context = 24
horizon = 6
input_size = 3

model = TransformerModel(
    input_size=input_size,
    d_model=128,
    nhead=4,
    
    num_layers=2,
    horizon=horizon
)

model.load_state_dict(torch.load("models/transformer_model.pth", map_location="cpu"))
model.eval()

print("✅ Model loaded")

example_input = torch.randn(1, context, input_size)

# ✅ TRACE (not script)
traced_model = torch.jit.trace(model, example_input)

print("🚀 Converting to CoreML...")

mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)],
    compute_units=ct.ComputeUnit.ALL
)

# mlmodel.save("models/transformer.mlmodel")
mlmodel.save("models/transformer.mlpackage")

print("✅ CoreML model saved")