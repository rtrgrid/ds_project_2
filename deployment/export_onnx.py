import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.transformer_model import TransformerModel

# config
context = 24
input_size = 3
horizon = 6

model = TransformerModel(input_size, 128, 4, 2, horizon)
model.load_state_dict(torch.load("models/transformer_model.pth", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, context, input_size)

print("🚀 Exporting ONNX...")

torch.onnx.export(
    model,
    dummy_input,
    "models/transformer.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=14
)

dynamic_axes={
    "input": {0: "batch_size"},
    "output": {0: "batch_size"}
}

print("✅ ONNX Exported")