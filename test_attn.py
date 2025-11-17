import os, torch
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available():", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())