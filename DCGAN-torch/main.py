from torch.optim import Adam
from torch.utils.data import Dataloader
from data import GAN_Dataset
from network import Generator, Discriminator
from train_test import train

###################
# HYPERPARAMETERS #
###################
BATCH_SIZE = 32
LEAKY_SLOPE = 0.2
IMG_SIZE = 32
SCALE = 16
LATENT = 128
###################

path = "TEMP"
data = GAN_Dataset(d_size=32, path=path)
dataloader = Dataloader(data, batch_size=BATCH_SIZE)
generator = Generator()
discriminator = Discriminator()
G_optim = Adam()
D_optim = Adam()

train(dataloader, generator, discriminator, G_optim, D_optim)


