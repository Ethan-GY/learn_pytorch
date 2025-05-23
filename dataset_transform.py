import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
    ])
trainset = torchvision.datasets.CIFAR10(root='./dataset2', train=True, transform=dataset_transform, download=True)
testset = torchvision.datasets.CIFAR10(root='./dataset2', train=False, transform=dataset_transform, download=True)

# print(testset[0])
writer = SummaryWriter("p10")
for i in range(10):
    img,target = testset[i]
    writer.add_image("test_set", img, i)
writer.close()