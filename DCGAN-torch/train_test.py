import torch
from loss import *
from main import BATCH_SIZE, LATENT

def train_step(X, generator, discriminator, gen_optim, disc_optim):
    noise = torch.randn(size=[BATCH_SIZE, LATENT])
    generated_images = generator(noise)
    real_output = discriminator(X)
    fake_output = discriminator(generated_images)

    gen_loss = generator_minimize_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)

    gen_optim.zero_grad()
    disc_optim.zero_grad()

    gen_loss.backward()
    disc_loss.backward()

    gen_optim.step()
    disc_optim.step()

    return gen_loss, disc_loss

def train(dataloader, generator, discriminator, gen_optim, disc_optim):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    size = len(dataloader.dataset)
    generator.train()
    discriminator.train()

    for epoch, X in enumerate(dataloader):
        X.to(device)
        #statistic functions

        train_step(X, generator, discriminator, gen_optim, disc_optim)

        #Printout and saving

    return