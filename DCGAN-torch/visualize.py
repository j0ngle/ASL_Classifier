from network import Generator
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

model_path = 'DCGAN-torch/models/model_e50.pt'

# model = Generator()
model = torch.load(model_path, map_location=torch.device('cpu'))

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
model.to(device)


def create_new_image():
    latent = 100
    fixed_noise = torch.randn(1, latent, 1, 1, device=device)

    model.eval()
    out = model(fixed_noise).squeeze().detach().cpu()
    out = (out * 127.5 + 127.5) / 255.

    noise = fixed_noise.squeeze().cpu()
    noise = noise.reshape(10, 10)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(noise)
    ax2.imshow(out.permute(1, 2, 0))
    ax1.axis('off')
    ax2.axis('off')
    fig.suptitle("Close window to generate new image!")
    plt.show()

while (True):
    create_new_image()
    

# model_path = 'models/model_e50.pt'

# # model = Generator()
# model = torch.load(model_path)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)

# latent = 100
# fixed_noise = torch.randn(1, latent, 1, 1, device=device)

# model.eval()
# out = model(fixed_noise).squeeze().detach().cpu()
# out = (out * 127.5 + 127.5) / 255.

# noise = fixed_noise.squeeze().cpu()
# noise = noise.reshape(10, 10)

# fig, (ax1, ax2) = plt.subplots(1, 2)

# ax1.imshow(noise)
# ax2.imshow(out.permute(1, 2, 0))
# ax1.axis('off')
# ax2.axis('off')
# button = Button(ax2, "Create new!")
# plt.show()