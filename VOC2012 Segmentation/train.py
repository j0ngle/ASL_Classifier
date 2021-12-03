import torch
from torch import functional as F
import torchvision
from data import *
from network import *

def loss_fn(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

def train(device, dataloader, model, optimizer, epochs):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        #Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        #Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    print("Done!")

def test(device, dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0

    with torch.no_grad:
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    num_epochs = 5
    lr = 0.001
    wd = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    PATH = '/home/wvuvl01/Documents/data/VOCdevkit/VOC2012'

    net = assemble_network()

    crop = (320, 480)
    train_dataloader = VOC_Dataset(True, crop, PATH)
    test_dataloader = VOC_Dataset(False, crop, PATH)
    op = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(device, train_dataloader, net, op)
        test(device, test_dataloader, net)

    print("Done!")