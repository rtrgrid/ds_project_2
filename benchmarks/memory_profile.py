import torch
import psutil
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.transformer_model import TransformerModel

process = psutil.Process(os.getpid())

def get_memory():
    return process.memory_info().rss / 1024**2  # MB

print("Initial Memory:", get_memory(), "MB")

model = TransformerModel(3, 128, 4, 2, 6)
model.load_state_dict(torch.load("models/transformer_model.pth", map_location="cpu"))

print("After Model Load:", get_memory(), "MB")

x = torch.randn(1, 24, 3)

with torch.no_grad():
    _ = model(x)

print("After Inference:", get_memory(), "MB")