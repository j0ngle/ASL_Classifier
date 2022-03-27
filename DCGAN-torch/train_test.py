import torch
from loss import *
from network import BATCH_SIZE, LATENT
from helpers import save_graph

def train_step(X, generator, discriminator, gen_optim, disc_optim):
    noise = torch.randn(size=[BATCH_SIZE, LATENT])

    ######################
    ##Update Discriminator
    ######################

    #On real images
    disc_optim.zero_grad()

    real_output = discriminator(X)

    real_loss = -1 * torch.mean(torch.log(real_output))
    real_loss.backward()
    D_x = real_output.mean().item()
    
    #On fake images
    generated_images = generator(noise)
    fake_output = discriminator(generated_images.detach())

    fake_loss = -1 * torch.mean(torch.log(1 - fake_output))
    fake_loss.backward()
    D_G_z1 = fake_output.mean().item()

    disc_loss = real_loss + fake_loss
    disc_optim.step()

    ###################
    ##Update Generator
    ###################
    gen_optim.zero_grad()

    fake_output = discriminator(generated_images)   #Create new fake_out so gradients can be updated
    gen_loss = -1 * torch.mean(torch.log(fake_output)) 

    gen_loss.backward()
    gen_optim.step()
    D_G_z2 = fake_output.mean().item()
    
    return gen_loss, disc_loss, D_x, (D_G_z1, D_G_z2)

def train(epochs, dataloader, generator, discriminator, gen_optim, disc_optim):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    size = len(dataloader)
    generator.train()
    discriminator.train()

    G_losses = []
    D_losses = []

    for epoch in range(epochs):
        for batch, X in enumerate(dataloader):
            X.to(device)
            Gl, Dl, D_x, D_G_z= train_step(X, generator, discriminator, gen_optim, disc_optim)

            G_losses.append(Gl.item())
            D_losses.append(Dl.item())

            if batch % 10 == 0:
                print("[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f, %.4f"
                        % (epoch, epochs, batch, size, Dl.item(), Gl.item(), D_x, D_G_z[0], D_G_z[1]))

        save_graph("G and D loss", "Iterations", "Loss", epoch, G_losses, "G", D_losses, "D")
            

    return