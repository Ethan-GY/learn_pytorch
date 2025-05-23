import torchvision.datasets

from torch.utils.data import DataLoader

from model import *
# 详细注释见train_gpu里
train_data = torchvision.datasets.CIFAR10(root='./dataset2', train=True, transform=torchvision.transforms.ToTensor(), download = False)
test_data = torchvision.datasets.CIFAR10(root='./dataset2', train=False, transform=torchvision.transforms.ToTensor(), download = False)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集的长度为{}".format(train_data_size))

train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)

gaoe = Gaoe()

loss_func = nn.CrossEntropyLoss()

learning_rate = 0.01

optimizer = torch.optim.SGD(gaoe.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 10

for i in range(epoch):
    print("-------第{}轮训练-------".format(i+1)) #i+1是因为range()函数的起始值为0
    for data in train_data_loader:
        img, label = data
        output = gaoe(img)
        loss = loss_func(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step,loss.item()))

    total_test_loss = 0
    with torch.no_grad():
        for data in test_data_loader:
            img, label = data
            output = gaoe(img)
            loss = loss_func(output, label)
            total_test_loss += loss.item()

    print('测试集整体loss:{}'.format(total_test_loss))

    torch.save(gaoe,'gaoe_{}.pth'.format(i+1))


