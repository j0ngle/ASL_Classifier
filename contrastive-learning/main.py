import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torch.optim import Adam
from data import Img_Dataset
from data import rand_aug
from network import Identity, Head
from loss import nt_xent

datapath = 'C:/Users/jthra/OneDrive/Documents/data/PetImages'

print("Loading dataset...")
data = Img_Dataset(datapath, num_per_class=10000)
dataloader = DataLoader(data, batch_size=32)
print("Dataset loaded!")

# encoder = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
encoder = resnet18(pretrained=True)
encoder.fc = Identity()
head = Head()
if torch.cuda.is_available():
    print("Sending models to CUDA device...")
    encoder.cuda()
    head.cuda()

encoder_optimizer = torch.optim.Adam(encoder.parameters())
head_optimizer = torch.optim.Adam(head.parameters())

device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


# TRAINING LOOP
epochs = 10
for i in range(epochs):
    for batch, X in enumerate(dataloader):
        X1 = rand_aug(X).to(device)
        X2 = rand_aug(X).to(device)

        encoder_optimizer.zero_grad()
        head_optimizer.zero_grad()

        embeddings_X1 = encoder(X1)
        embeddings_X2 = encoder(X2)
        output_X1 = head(embeddings_X1)
        output_X2 = head(embeddings_X2)

        loss = nt_xent(output_X1, output_X2)
        loss.backward()

        encoder_optimizer.step()
        head_optimizer.step()

        if batch % 50 == 0:
            print(f'(Batch: {batch}) [{i}/{epochs}] Loss: {loss}')
        

