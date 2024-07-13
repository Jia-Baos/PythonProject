import torch
import torch.nn as nn


class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        print("ResNetBasicBlock 1: ", output.size())

        output = self.conv2(output)
        output = self.bn2(output)
        print("ResNetBasicBlock 2: ", output.size())

        output = self.relu(x + output)
        print("ResNetBasicBlock 3: ", output.size())

        return output


class ResNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetDownBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.extra = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        print("ResNetDownBlock 1: ", output.size())

        output = self.conv2(output)
        output = self.bn2(output)
        print("ResNetDownBlock 2: ", output.size())

        output = self.relu(extra_x + output)
        print("ResNetDownBlock 3: ", output.size())

        return output


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(ResNetBasicBlock(64, 64, 1),
                                    ResNetBasicBlock(64, 64, 1))
        self.layer2 = nn.Sequential(ResNetDownBlock(64, 128, [2, 1]),
                                    ResNetBasicBlock(128, 128, 1))
        self.layer3 = nn.Sequential(ResNetDownBlock(128, 256, [2, 1]),
                                    ResNetBasicBlock(256, 256, 1))
        self.layer4 = nn.Sequential(ResNetDownBlock(256, 512, [2, 1]),
                                    ResNetBasicBlock(512, 512, 1))

        # 如此可以固定全连接层的输入
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        output = self.conv1(x)
        print("first conv: ", output.size())
        output = self.bn1(output)
        output = self.maxpool(output)
        print("first maxpool: ", output.size())

        output = self.layer1(output)
        print("first Resnetblock: ", output.size())
        output = self.layer2(output)
        print("second Resnetblock: ", output.size())
        output = self.layer3(output)
        print("third Resnetblock: ", output.size())
        output = self.layer4(output)
        print("fourth Resnetblock: ", output.size())

        output = self.avgpool(output)
        output = output.reshape(x.shape[0], -1)
        print("ready enter fc: ", output.size())

        output = self.fc(output)
        print("output of fc: ", output.size())
        return output


if __name__ == '__main__':
    x = torch.rand((3, 1, 224), dtype=torch.float)
    print(x.size())
    model = ResNet18()
    y = model(x)
    print(y.size())
