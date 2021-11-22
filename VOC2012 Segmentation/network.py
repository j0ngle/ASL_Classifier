import torch
import torch.nn as nn
import torch.functional as F
import torchvision
from torch.utils.data import Dataset, Dataloader
import matplotlib.pyplot as plt

class FCN(nn.Module):
    def __init__(self):
        self.convstack = nn.Sequential([
            nn.Conv2d(3, 32, 5, stride=2)
        ])

        self.transpose = nn.Sequential([
            nn.ConvTranspose2d()
        ])

    
    def forward(self):
        return 0