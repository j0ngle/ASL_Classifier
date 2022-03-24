from torch.optim import Adam
from torch.utils.data import DataLoader
from data import GAN_Dataset
from network import Generator, Discriminator
from network import BATCH_SIZE
from train_test import train



path = "C:/Users/jthra/OneDrive/Documents/data/img_align_celeba"

print("Loading dataset...")
data = GAN_Dataset(d_size=32, path=path)
dataloader = DataLoader(data, batch_size=BATCH_SIZE)
print("Dataset loaded!")

generator = Generator()
discriminator = Discriminator()
G_optim = Adam(generator.parameters())
D_optim = Adam(discriminator.parameters())
epochs = 10

for epoch in range(epochs):
    print("Starting epoch {}...".format(epoch+1))
    train(dataloader, generator, discriminator, G_optim, D_optim)


