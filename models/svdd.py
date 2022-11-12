import torch
import torch.nn as nn

class SVDD(nn.Module):
    def __init__(self, c, r):
        super(SVDD, self).__init__()
        self.C = nn.Parameter(c.view(1, -1))
        self.R = nn.Parameter(r)
        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, x):
        return  torch.sigmoid(self.pdist(x, self.C) - self.R)

class SVDD_Plus(nn.Module):
    def __init__(self, dim):
        super(SVDD_Plus, self).__init__()
        self.C = nn.Parameter(torch.randn(1, dim))
        self.R = nn.Parameter(torch.randn(1, 1))
        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, x):
        return  torch.sigmoid(self.pdist(x, self.C) - self.R).view(-1)