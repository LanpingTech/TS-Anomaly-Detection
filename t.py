import torch
import torch.nn as nn

x = torch.ones(2, 4)
y = torch.ones(1, 4) *3
print(x - y)