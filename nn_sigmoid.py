import torch
import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset2", train = False, download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class gaoe(torch.nn.Module):
    def __init__(self):
        super(gaoe, self).__init__()
        # self.relu1 =  torch.nn.ReLU()
        self.sigmoid1 = torch.nn.Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

Gaoe = gaoe()

writer = SummaryWriter("logs_sigmoid")
step = 0

for data in dataloader:
    imgs,target = data
    writer.add_images("input", imgs, step)
    output = Gaoe(imgs)
    writer.add_images("output", output, step)
    step += 1
writer.close()

