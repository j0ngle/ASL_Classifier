import torch
from loss import *
from network import BATCH_SIZE, LATENT

def train_step(X, generator, discriminator, gen_optim, disc_optim, device):
    noise = torch.randn(size=[BATCH_SIZE, LATENT])

    #Update D
    disc_optim.zero_grad()

    real_output = discriminator(X)
    
    generated_images = generator(noise)
    fake_output = discriminator(generated_images.detatch())

    disc_loss = discriminator_loss(real_output, fake_output)
    disc_loss.backward()

    disc_optim.step()

    #Update G
    gen_optim.zero_grad()
    gen_loss = generator_minimize_loss(fake_output)

    gen_loss.backward()

    gen_optim.step()
    
    return gen_loss, disc_loss

def train(dataloader, generator, discriminator, gen_optim, disc_optim):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    size = len(dataloader.dataset)
    generator.train()
    discriminator.train()

    for batch, X in enumerate(dataloader):
        #statistic functions
        X.to(device)
        train_step(X, generator, discriminator, gen_optim, disc_optim, device)

        #Printout and saving

    return