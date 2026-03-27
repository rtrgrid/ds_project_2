import torch

class CircularBuffer:
    def __init__(self, size, feature_dim):
        self.buffer = torch.zeros(size, feature_dim)
        self.size = size
        self.ptr = 0
        self.full = False

    def append(self, x):
        self.buffer[self.ptr] = x
        self.ptr = (self.ptr + 1) % self.size

        if self.ptr == 0:
            self.full = True

    def get(self):
        # ✅ NO CONCAT → just roll (fast)
        return torch.roll(self.buffer, -self.ptr, dims=0)