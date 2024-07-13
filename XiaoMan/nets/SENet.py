import torch
import torch.nn as nn


class SELayer(nn.Module):
    """
    The mechanism of SELayer is giving every channel a attention weight, so the output of SELayer is the input multiple attention weights
    the Squeeze and Excitation realizes the interaction of features from different channels
    the activation func of Sigmoid realizes the flexibility of weights, which means several channels can embrace the high weights at the same time
    """
    def __init__(self, channel=3, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        print("avg_pool: ", y.size())
        y = self.fc(y).view(b, c, 1, 1)
        print("Squeeze and Excitation: ", y.size())
        y = y.expand_as(x)
        print("expand_as x: ", y.size())
        return x * y


if __name__ == '__main__':
    x = torch.rand((4, 3, 4, 4), dtype=torch.float)
    print("input's size: ", x.size())
    model = SELayer()
    y = model(x)
    print(y.size())
    print("output's size: ", y.size())
