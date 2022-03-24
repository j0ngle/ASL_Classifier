import torch
from torch import nn

###################
# HYPERPARAMETERS #
###################
BATCH_SIZE = 128
LEAKY_SLOPE = 0.2
IMG_SIZE = 64
SCALE = 16
LATENT = 64 #nz
F_MAPS = 64 #ngf/d
###################

scaled_size = IMG_SIZE // 16

#TODO: Initialize weights
def conv_transpose(in_channels, out_channels, k_size=5, stride=2, padding=0, bias=False):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, stride=stride, 
                            kernel_size=k_size, padding=padding, bias=bias),
        nn.ReLU(inplace=True)
    )

#TODO: Initialize weights
def conv(in_channels, out_channels, k_size=5, stride=2, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=k_size, padding=padding, bias=bias),
        nn.LeakyReLU(LEAKY_SLOPE, inplace=True)
    )

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.dense   = nn.Linear(LATENT, scaled_size*scaled_size*LATENT)
        self.dropout = nn.Dropout()
        self.convt_1 = conv_transpose(LATENT, F_MAPS*8, k_size=5, stride=2)
        self.convt_2 = conv_transpose(F_MAPS*8, F_MAPS*4, k_size=5, stride=2, padding=1)
        self.convt_3 = conv_transpose(F_MAPS*4, F_MAPS*2, k_size=5, stride=2, padding=1)
        self.convt_4 = conv_transpose(F_MAPS*2, F_MAPS, k_size=5, stride=2, padding=1)
        self.out     = nn.ConvTranspose2d(F_MAPS, 3, stride=1, kernel_size=5, padding=1)
        self.tanh    = nn.Tanh()

    def forward(self, x):
        x = self.dense(x)
        x = torch.reshape(x, [BATCH_SIZE, F_MAPS, scaled_size, scaled_size])

        x = self.convt_1(x)
        x = self.convt_2(x) # <- 4x4
        x = self.convt_3(x) # <- 8x8
        x = self.convt_4(x) # <- 16x16
        x = self.out(x)     # <- 32x32
        return self.tanh(x) # <- 64x64

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_0  = nn.Conv2d(3, F_MAPS, stride=2, kernel_size=5, padding=1)
        self.leaky   = nn.LeakyReLU(LEAKY_SLOPE)
        self.conv_1  = conv(F_MAPS, F_MAPS*2, k_size=5, stride=2)
        self.conv_2  = conv(F_MAPS*2, F_MAPS*4, k_size=5, stride=2)
        self.conv_3  = conv(F_MAPS*4, F_MAPS*8, k_size=5, stride=2)
        self.flatten = nn.Flatten()
        self.dense   = nn.Linear(4*4*F_MAPS*8, 1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_0(x)  #<- 64x64
        x = self.leaky(x)
        x = self.conv_1(x)  #<- 32x32
        x = self.conv_2(x)  #<- 16x16
        x = self.conv_3(x)  #<- 8x8
        # x = self.flatten(x) # <-4x4
        # x = self.dense(x)   #sigmoid activation
        return self.Sigmoid(x)
