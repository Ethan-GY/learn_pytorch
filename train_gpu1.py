import torchvision.datasets

from torch.utils.data import DataLoader

import torch
from torch import nn

# from model import *

#准备数据
train_data = torchvision.datasets.CIFAR10(root='./dataset2', train=True, transform=torchvision.transforms.ToTensor(), download = False)
test_data = torchvision.datasets.CIFAR10(root='./dataset2', train=False, transform=torchvision.transforms.ToTensor(), download = False)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集的长度为{}".format(train_data_size))

#创建数据加载器
train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)

#定义模型
class Gaoe(nn.Module):
    def __init__(self):
        super(Gaoe, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    gaoe = Gaoe()
    input = torch.ones((64, 3, 32, 32))
    output = gaoe(input)
    print(output.shape)

#创建模型
gaoe = Gaoe()
if torch.cuda.is_available():
    gaoe = gaoe.cuda()

#定义损失函数和优化器
loss_func = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_func = loss_func.cuda()

learning_rate = 0.01

optimizer = torch.optim.SGD(gaoe.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 10

for i in range(epoch):
    print("-------第{}轮训练-------".format(i+1)) #i+1是因为range()函数的起始值为0

    #训练步骤开始
    gaoe.train() #打开训练模式，可有可无
    for data in train_data_loader:
        img, label = data
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        output = gaoe(img)
        loss = loss_func(output, label)

        #优化器梯度清零
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step,loss.item()))

    #测试步骤开始
    gaoe.eval() #测试模式，可有可无
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():#不计算梯度
        for data in test_data_loader:
            img, label = data
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()
            output = gaoe(img)
            loss = loss_func(output, label)
            total_test_loss += loss.item()
            accuracy = (output.argmax(dim=1) == label).sum()
            total_accuracy += accuracy.item()

    print('测试集准确率:{}'.format(total_accuracy / test_data_size))
    print('测试集整体loss:{}'.format(total_test_loss))
    total_test_step += 1

    torch.save(gaoe,'gaoe_{}.pth'.format(i+1))


