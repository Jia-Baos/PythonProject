import torch
from torch import nn


# 定义网络模型
class MyNet(nn.Module):
    # 初始化网络
    def __init__(self):
        super(MyNet, self).__init__()

        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.Sigmoid = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.f6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def forward(self, x):
        x = self.Sigmoid(self.c1(x))
        x = self.s2(x)
        x = self.Sigmoid(self.c3(x))
        x = self.s4(x)
        x = self.c5(x)
        x = x.view(x.size()[0], -1)  # Pytorch1.1.0没有nn.flatten()函数，所以自己修改了
        x = self.f6(x)
        x = self.output(x)
        return x


class MyNet1(nn.Module):
    # 初始化网络
    def __init__(self):
        super(MyNet1, self).__init__()

        self.c1 = nn.Conv2d(in_channels=1, out_channels=31, kernel_size=3)
        self.Sigmoid = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.f6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def forward(self, x):
        x = self.Sigmoid(self.c1(x))
        x = self.s2(x)
        x = self.Sigmoid(self.c3(x))
        x = self.s4(x)
        x = self.c5(x)
        x = x.view(x.size()[0], -1)  # Pytorch1.1.0没有nn.flatten()函数，所以自己修改了
        x = self.f6(x)
        x = self.output(x)
        return x


if __name__ == "__main__":
    x = torch.rand([1, 1, 28, 28])
    model = MyNet()
    y = model(x)
