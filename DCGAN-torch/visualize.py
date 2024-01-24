from network import Generator
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

model_path = 'models/model_e50.pt'

# model = Generator()
model = torch.load(model_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


def create_new_image(val):
    print("here")
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
    plt.show()

    


# create_new_image(0)
while (True):
    axes = plt.axes([0.5, 0.5, 0.5, 0.5])
    bnext = Button(axes, 'Add',color="yellow")
    bnext.on_clicked(create_new_image)
    plt.show()

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