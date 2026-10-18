import platform
import sys

import numpy as np
import torch
from torch import nn


def main():
    print("Python:", sys.version.split()[0])
    print("Interpreter:", sys.executable)
    print("Architecture:", platform.machine())
    print("NumPy:", np.__version__)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("MPS available:", torch.backends.mps.is_available())

    # Start with CPU to verify the basic environment.
    torch.manual_seed(42)
    model = nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    inputs = torch.randn(4, 3)
    targets = torch.randn(4, 1)

    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(inputs), targets)
    loss.backward()

    assert torch.isfinite(loss).item()
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad).all().item()

    optimizer.step()
    print("Forward pass, backward pass, and optimizer step succeeded.")


if __name__ == "__main__":
    main()