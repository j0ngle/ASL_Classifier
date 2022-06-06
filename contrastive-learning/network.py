from sklearn.cluster import k_means
from sklearn.preprocessing import KernelCenterer
from torch import nn

LEAKY_SLOPE = 0.2

def conv(in_channels, out_channels, k_size=5, stride=2, padding=0, bias=False, bn=True):

    if bn:
        layers = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=k_size, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(LEAKY_SLOPE, inplace=True)
    )

    else:
        layers = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=k_size, padding=padding, bias=bias),
        nn.LeakyReLU(LEAKY_SLOPE, inplace=True)
    )

    return layers

# TODO: Change this to ResNet50 as encoder
# Then make custom dense network
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.downblock = nn.Sequential(
            conv(3, 32, k_size=4, stride=2, padding=1, bn=False),
            conv(32, 64, k_size=4, stride=2, padding=1),
            conv(64, 128, k_size=4, stride=2, padding=1),
            conv(128, 256, k_size=4, stride=2, padding=1),
            conv(256, 512, k_size=4, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(8*8*512, 10000),
            nn.Linear(10000, 1000),
            nn.Linear(1000, 100)
        )

    def forward(self, x):
        return self.downblock(x)