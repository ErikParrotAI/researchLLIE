import torch
print("CUDA?", torch.cuda.is_available())
print("GPU  :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print(torch.__version__, torch.version.cuda)