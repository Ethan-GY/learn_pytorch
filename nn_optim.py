import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 其他需要import看💡提示，一键导入即可
dataset = torchvision.datasets.CIFAR10("../dataset2", train=False, download=False,
                                        transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,
                        batch_size=1)  # batchsize是每次选的个数,其他有些不常用的参数，比如shuffle下一次选取数据时是否打乱顺序,droplast是把最后不全的一组选择去掉

class Gaoe(torch.nn.Module):
    def __init__(self):
        super(Gaoe, self).__init__()  # 可以直接用code generation的重写功能填充

        self.model1 = torch.nn.Sequential( # 这里用Sequential可以直接把多个层组合起来,方便后面直接调用x = self.model1(x)
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1,
                            padding=2),  # command+p提示具体填哪些参数。其中kernelsize是过滤器大小3*3，会默认一开始随机初始化，stride可理解为步长，padding是边缘填充
            torch.nn.MaxPool2d(kernel_size=2, ceil_mode=True),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1,
                            padding=2),
            torch.nn.MaxPool2d(kernel_size=2, ceil_mode=True),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1,
                            padding=2),
            torch.nn.MaxPool2d(kernel_size=2, ceil_mode=True),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=1024, out_features=64),
            torch.nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


gaoe = Gaoe()  # 创建实例
# print(gaoe)
#
# writer = SummaryWriter("logs")  # 建立log日志

loss = torch.nn.CrossEntropyLoss()  # 定义损失函数
optimizer = torch.optim.SGD(gaoe.parameters(), lr=0.01)  # 定义优化器
for epoch in range(20):  # 训练20轮
    running_loss = 0.0
    for data in dataloader:
        imgs, target = data  # data中有图像自身和它对应的标签
        output = gaoe(imgs)
        # print(output)
        # print(target)
        result_loss = loss(output, target)
        optimizer.zero_grad()  # 梯度清零
        result_loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        running_loss += result_loss.item()   # 累加loss值
    print(running_loss)
