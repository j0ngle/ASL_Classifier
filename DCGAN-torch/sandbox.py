from data import *
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from helpers import *
from torch import nn
import torch
import logging
from torchvision.models import inception_v3

inception = inception_v3(pretrained=True)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

inception.fc = Identity()
inception.dropout = Identity()

print(inception)
x = torch.randn(64, 3, 299, 299)
output = inception(x)
print(output.logits)