import torchvision
from PIL import Image

import torch
from torch import nn


image_path = "images/dog.jpg"
image = Image.open(image_path)
image = image.convert("RGB")

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)

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

model = torch.load("gaoe_30.pth", weights_only=False, map_location=torch.device('cpu'))
# print(model)

image = torch.reshape(image, (1, 3, 32, 32))  # reshape image to (batch_size, channels, height, width)
model.eval()
with torch.no_grad(): #良好代码习惯，不计算梯度，减小运算量
    output = model(image)
print(output)
print(output.argmax(1))