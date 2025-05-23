import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from test_tb import writer

testset = torchvision.datasets.CIFAR10(root='./dataset2', train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(testset, batch_size=64)

class gaoe(nn.Module):
    def __init__(self):
        super(gaoe, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3,1,0)

    def forward(self,x):
        x = self.conv1(x)
        return x

Gaoe = gaoe()

step = 0
for data in dataloader:
    imgs,targets = data
    output = Gaoe(imgs)
    # print(imgs.shape)
    # print(output.shape)

    writer.add_images("input",imgs,step)

    output = torch.reshape(output,(-1,3,30,30))

    writer.add_images("output",output,step)

    step += 1

writer.close()