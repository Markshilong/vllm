import torch
a = torch.randn(768, 3072, device='cuda', dtype=torch.float16)
b = torch.randn(3072, 4, device='cuda', dtype=torch.float16)
torch.matmul(a, b)
