import torch
from torch import nn

LEAKY_SLOPE = 0.2
IMG_SIZE = 32
SCALE = 16
LATENT = 128

scaled_size = IMG_SIZE // 16

#TODO: Initialize weights
def conv_transpose(in_channels, out_channels, k_size, stride):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, stride=stride, kernel_size=k_size),
        nn.ReLU()
    )

#TODO: Initialize weights
def conv(in_channels, out_channels, k_size, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=k_size),
        nn.LeakyReLU(LEAKY_SLOPE)
    )

class generator(nn.Module):
    def __init__(self):
        self.dense   = nn.Linear(LATENT, scaled_size*scaled_size*LATENT)
        self.dropout = nn.Dropout()
        self.convt_1 = conv_transpose(1024, 512, k_size=5, stride=2)
        self.convt_2 = conv_transpose(512, 256, k_size=5, stride=2)
        self.convt_3 = conv_transpose(256, 128, k_size=5, s_size=2)
        self.out     = nn.ConvTranspose2d(128, 3, stride=1, kernel_size=5)
        self.tanh    = nn.Tanh()

    def forward(self, x):
        x = self.dense(x)
        x = torch.reshape(x, [scaled_size, scaled_size, 128])
        x = self.convt_1(x)
        x = self.convt_2(x)
        x = self.convt_3(x)
        x = self.out(x)
        return self.tanh(x)

class discriminator(nn.Module):
    def __init__(self):
        self.conv_0  = nn.Conv2d(3, 128, stride=2, kernel_size=5)
        self.leaky   = nn.LeakyReLU(LEAKY_SLOPE)
        self.conv_1  = conv(128, 256, k_size=5, stride=2)
        self.conv_2  = conv(256, 512, k_size=5, stride=2)
        self.conv_3  = conv(512, 1024, k_size=5, stride=2)
        self.flatten = nn.Flatten()
        self.dense   = nn.Linear(scaled_size*scaled_size*1024, 1)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.leaky(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.flatten(x)
        return self.dense(x)    #sigmoid activation
