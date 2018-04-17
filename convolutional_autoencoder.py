__author__ = 'SherlockLiao'

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os

import LungsDataset as ld

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 128, 128)
    return x

num_epochs = 100
batch_size = 32
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ld.LungsDataset('./lungs', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True)
            # nn.MaxPool2d(2, stride=1)
        )

        self.encoderLinear = nn.Sequential(
            nn.Linear(8*8*1024, 4096)
        )

        self.decoderLinear = nn.Sequential(
            nn.Linear(4096, 8 * 8 * 1024)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        # print(x)
        # x = x.view(x.size(0), 8 * 8 * 1024)
        # x = self.encoderLinear(x)
        # x = self.decoderLinear(x)
        # x = x.view(x.size(0), 1024, 8, 8)
        x = self.decoder(x)
        return x


# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

use_gpu = False # torch.cuda.is_available()

model = autoencoder()

if use_gpu:
    model = model.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

for epoch in range(num_epochs):
    i = 0
    for data in dataloader:
        img = data
        if use_gpu:
            img = Variable(img).cuda()
        else:
            img = Variable(img)
        # img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("batch")

        if i % 30 == 0:
            print('epoch {}, iter {}, loss:{:.4f}'
                  .format(epoch + 1, i, loss.data[0]))

        i += 1

    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.data[0]))

    pic = to_img(output.cpu().data)
    save_image(pic, './dc_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './conv_autoencoder.pth')
