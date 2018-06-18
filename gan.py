import torch
import os

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torchvision.datasets import CIFAR10
from tqdm import tqdm


class Generator(nn.Module):
    def __init__(self, img_shape, input_size):
        super(Generator, self).__init__()

        self.init_size = img_shape // 4
        self.l1 = nn.Sequential(nn.Linear(input_size, 128*self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_shape // 2**4
        self.adv_layer = nn.Sequential(nn.Linear(128*ds_size**2, 1),
                                       nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


input_noise_size = 100
batch_size = 8
device = torch.device("cuda:0")
nb_epoch = 200

print("Loading data")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0, 0, 0), (1, 1, 1))])
trainset = CIFAR10(".", train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

print("Building Generator network")
gen = Generator(32, input_noise_size).to(device)
gen_optim = optim.Adadelta(gen.parameters())

print("Building Discriminator network")
discri = Discriminator(32).to(device)
discri_optim = optim.Adadelta(discri.parameters())

loss_function = nn.BCELoss()

k = 5

print("Training")

for epoch in range(nb_epoch):
    with tqdm(trainloader, bar_format="{l_bar}{bar}{n_fmt}/{total_fmt}, ETA:{remaining}{postfix}", ncols=80, desc="Epoch " + str(epoch)) as t:
        disc, gene = 0, 0
        n = 0
        for real_imgs, _ in t:
            real_imgs = real_imgs.to(device)
            valid = torch.ones(real_imgs.size(0)).to(device)
            fake = torch.zeros(real_imgs.size(0)).to(device)

            # Train Discriminator
            for _ in range(k):
                real_loss = loss_function(discri(real_imgs), valid)
                fake_loss = loss_function(discri(gen_imgs.detach()), fake)
                discri_loss = (real_loss + fake_loss) / 2

                discri_optim.zero_grad()
                discri_loss.backward()
                discri_optim.step()

            # Train Generator
            input_noise = torch.randn(valid.size(0), input_noise_size).to(device)
            gen_imgs = gen(input_noise)

            gen_loss = loss_function(discri(gen_imgs), valid)

            gen_optim.zero_grad()
            gen_loss.backward()
            gen_optim.step()



            n += 1
            disc = ((n-1) * disc + discri_loss.tolist()) / n
            gene = ((n-1) * gene + gen_loss.tolist()) / n
            t.set_postfix({"gen_loss": "{0:.3f}".format(gene), "discri_loss": "{0:.3f}".format(disc)})

        os.system("mkdir outputs/epoch" + str(epoch))
        r = torch.rand(10, input_noise_size).to(device)
        pred = gen(r)
        pred = pred.detach().cpu().numpy()
        pred = np.swapaxes(pred, 1, 3)
        pred = np.swapaxes(pred, 1, 2)
        pred = (pred + 1) / 2
        for i, p in enumerate(pred):
            plt.imsave("outputs/epoch" + str(epoch) + "/" + str(i) + ".png", p)
