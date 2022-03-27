from data import *
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from helpers import *
from torch import nn
import torch

epoch = 0
epochs = 10
batch = 0
size = 1000
Dl = .5
Gl = .2
D_x = .2
D_G_z = (1, 5)

print("[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f, %.4f"
                        % (epoch, epochs, batch, size, Dl, Gl, D_x, D_G_z[0], D_G_z[1]))