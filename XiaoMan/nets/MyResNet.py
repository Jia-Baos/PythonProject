import torch
import torch.nn as nn
from torch.nn import functional as F


class RestNetBasicBlock(nn.Module):
    """
    This block indicates the full line part in the ResNet
    """
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    """
    This block indicates the dotted line part in the ResNet
    """
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 旁支也需进行卷积
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class RestNet18(nn.Module):
    """
    The three layers are 1x1, 3x3 and 1x1 convolution, where the 1x1 layers are responsible for reducing and the increase(restoring) dinmensions(channels), leaving the 3x3 layer a bottleneck with smaller input/output dimensions.

    The layer 1x1 with stride 2 just like the dowmsample, which is easy to reduce the dimensions(channels) and computational complexity.
    
    A simple example to explain the mechanism of 1x1:
        1: 256 -> 3x3x256 -> 256
        parameters: 256x3x3x256 = 589824
        2: 256 -> 1x1x64 -> 3x3x64 -> 1x1x256 -> 256
        parameters: 256x1x1x64 + 64x3x3x64 + 64x1x1x256 = 69632
    The total number of parameters has reduced drastically.
    """
    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))

        # 此步可以理解为再次进行了下采样
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    # test MyResNet
    x = torch.rand((3, 224, 224), dtype=torch.float)
    x = x.unsqueeze(0)
    print(x.size())
    model = RestNet18()
    y = model(x)
    print(y)
