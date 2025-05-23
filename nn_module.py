import torch
from torch import nn


class gaoe(nn.Module):
    def __init__(self):
        super(gaoe, self).__init__()

    def forward(self, input):
        output = input +1
        return output

gaoe = gaoe()
x = torch.tensor(1.0)
output = gaoe(x)
print(output)