
import torch

# 1. 检查版本号 (应该包含 +cu124)
print(f"PyTorch Version: {torch.__version__}")

# 2. 检查 CUDA 是否可用 (必须是 True)
print(f"CUDA Available:  {torch.cuda.is_available()}")

# 3. 检查当前显卡 (应该能显示显卡名称)
if torch.cuda.is_available():
    print(f"GPU Name:        {torch.cuda.get_device_name(0)}")
else:
    print("Error: CUDA not detected!")