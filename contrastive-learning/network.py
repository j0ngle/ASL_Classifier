import torch
from torch import nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()

        self.downblock = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

    def forward(self, x):
        return self.downblock(x)

