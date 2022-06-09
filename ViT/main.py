import torch
from torch import nn
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from network import ViT

root = 'C:/Users/jthra/OneDrive/Documents/data'
training_data = CIFAR100(root=root, train=True, download=True, transform=ToTensor())
test_data = CIFAR100(root=root, train=False, transform=ToTensor())

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

model = ViT()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

device = "cuda" if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f'Using {device} device')

model.train()
epochs = 10
for epoch in range(epochs):
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(train_dataloader.dataset):>5d}]")
