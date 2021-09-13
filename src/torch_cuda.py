import torch

print(torch.__version__)
# 1.9.0
 
print(torch.cuda.is_available())
# True
print(torch.cuda.device_count())
# 1
print(torch.cuda.current_device())
# 0
print(torch.cuda.get_device_name())
# NVIDIA Tegra X1
