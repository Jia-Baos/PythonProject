import torch
import torch.nn as nn


class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.brach_avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1)
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=5, padding=2)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.Conv2d(24, 24, kernel_size=3, padding=1)
        )

    def forward(self, x):
        branch_pool = self.brach_avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)
        branch1 = self.branch1(x)
        branch5 = self.branch5(x)
        branch3 = self.branch3(x)

        # 在channels上拼接四个分支，要求所有channels的特征图大小一致
        return torch.cat((branch_pool, branch1, branch5, branch3), dim=1)


class GoogModel(nn.Module):
    def __init__(self):
        super(GoogModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)
        self.Inception1 = Inception(in_channels=10)
        self.Inception2 = Inception(in_channels=20)
        self.mp = nn.MaxPool2d(kernel_size=2)
        # self.fc = nn.Linear(53*53*88, 10)
        self.avgpool = nn.AdaptiveAvgPool2d((50, 50))
        self.fc = nn.Linear(50 * 50 * 88, 10)

    def forward(self, x):
        x = nn.ReLU(inplace=True)(self.mp(self.conv1(x)))
        x = self.Inception1(x)
        x = nn.ReLU(inplace=True)(self.mp(self.conv2(x)))
        x = self.Inception2(x)
        print("ready enter fcn: ", x.size())
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        v = self.fc(x)
        return v


if __name__ == '__main__':
    x = torch.rand((2, 3, 224, 224), dtype=torch.float)
    model = GoogModel()
    y = model(x)
    print(y.size())
