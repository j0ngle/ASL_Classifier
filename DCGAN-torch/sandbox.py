from data import *
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from helpers import *
from torch import nn
import torch

latent = 128
noise = torch.randn(size=[32, latent])
m = nn.Linear(latent, 2*2*latent)

output = m(noise)
print(output.size())

x = torch.reshape(output, [32, 2, 2, latent])
