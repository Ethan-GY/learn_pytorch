
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

testset = torchvision.datasets.CIFAR10(root='./dataset2', train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = DataLoader(testset, batch_size=64, shuffle=True, num_workers=0,drop_last=False)

# img,target = testset[0]
# print(img.shape)
# print(target)

writer = SummaryWriter("dataloader") #日志保存的路径名
step = 0
for data in test_loader:
    imgs, targets = data
    writer.add_images("testset", imgs, step)
    step += 1

writer.close()

