import torch
import torch.nn as nn
import torch.functional as F
import torchvision
from torch.utils.data import Dataset, Dataloader
import matplotlib.pyplot as plt

def get_encoder(net='resnet18'):
    if (net == 'resnet18'):
        net = torchvision.models.resnet18(pretrained=True)
        return nn.Sequential(*list(net.children())[:-2])
    
    raise Exception('NameError: invalid name provided for parameter "net"')

def assemble_network(encoder='resnet18'):
    network = get_encoder(net=encoder)

    #Add 1x1 convolution
    num_classes = 21
    network.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))

    #Add decoder
    network.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes, 
                                                        kernel_size=64, padding=16, stride=32))
                                                        
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight


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