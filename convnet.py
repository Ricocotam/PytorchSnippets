import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision.datasets import CIFAR10
from tqdm import tqdm


class ConvNet(nn.Module):
    """docstring for ConvNet."""
    def __init__(self, im_shape):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1),
                nn.ELU(),
                nn.Conv2d(8, 8, 3, padding=1),
                nn.ELU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, 3, padding=1),
                nn.ELU(),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ELU(),
                nn.MaxPool2d(2)
            )

        self.fc = nn.Sequential(
                nn.Dropout(.25),
                nn.Linear(16*8*8, 128),
                nn.ELU(),
                nn.Dropout(.25),
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Dropout(.25),
                nn.Linear(64, 10),
                nn.Softmax(1)
            )

    def forward(self, x):
        out = self.conv(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out




def train(model, nb_epoch, trainloader, optimizer, loss_function, device):
    for epoch in range(nb_epoch):
        with tqdm(trainloader, bar_format="{l_bar}{bar}{n_fmt}/{total_fmt}, ETA:{remaining}{postfix}", ncols=80, desc="Epoch " + str(epoch)) as t:
            mean_loss, n = 0, 0
            for x, y in t:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                loss = loss_function(pred, y)

                n += 1
                mean_loss += (n-1) * mean_loss + loss.tolist() / n
                t.set_postfix({"train_loss": "{0:.3f}".format(loss.tolist())})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


device = torch.device("cuda:0")
nb_epoch = 100

print("Loading data")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0, 0, 0), (1, 1, 1))])
trainset = CIFAR10(".", train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

print("Building network")
model = ConvNet((32, 32)).to(device)
#model = LeNet().to(device)
optimizer = optim.Adadelta(model.parameters())
loss_function = nn.CrossEntropyLoss()

print("Training")
train(model, nb_epoch, trainloader, optimizer, loss_function, device)
