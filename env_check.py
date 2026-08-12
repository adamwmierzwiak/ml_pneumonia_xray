import torch
import sys
import random

print(f"Wersja Pythona: {sys.version}")
print(f"Wersja PyTorch: {torch.__version__}")

# Sprawdzamy dostępność GPU
cuda_available = torch.cuda.is_available()
print(f"Czy CUDA jest dostępna? {cuda_available}")

if cuda_available:
    print(f"Urządzenie: {torch.cuda.get_device_name(0)}")
else:
    print("Urządzenie: CPU (Docker na macOS nie widzi GPU)")

# Prosty test obliczeniowy
x = torch.rand(5, 3)
print("\nTestowy tensor:\n", x)

random.seed(123)
print()

randoms = []

for _ in range(5):
    randoms.append(random.random())

print(randoms)