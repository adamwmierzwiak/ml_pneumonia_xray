import random
import sys

import numpy
import pandas
import sklearn
import torch
import torchmetrics
import torchvision

print(f"Python version: {sys.version}")
print(f"torch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"torchmetrics: {torchmetrics.__version__}")
print(f"numpy: {numpy.__version__}")
print(f"pandas: {pandas.__version__}")
print(f"scikit-learn: {sklearn.__version__}")

# Check GPU availability
cuda_available = torch.cuda.is_available()
print(f"\nCUDA available? {cuda_available}")

if cuda_available:
    print(f"Device: {torch.cuda.get_device_name(0)}")
else:
    print("Device: CPU (Docker on macOS has no GPU access)")

# Simple compute smoke test
x = torch.rand(5, 3)
print("\nTest tensor:\n", x)

random.seed(123)
print()

randoms = []

for _ in range(5):
    randoms.append(random.random())

print(randoms)
