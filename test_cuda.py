import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device selected: {device}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device count: {torch.cuda.device_count()}")
else:
    print("CUDA not available - using CPU")
