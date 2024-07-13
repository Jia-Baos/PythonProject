import torch
import torch.nn as nn


class SELayer(nn.Module):
    """
    The mechanism of SELayer is giving every channel a attention weight, so the output of SELayer is the input multiple attention weights
    the Squeeze and Excitation realizes the interaction of features from different channels
    the activation func of Sigmoid realizes the flexibility of weights, which means several channels can embrace the high weights at the same time
    """

    def __init__(self, in_channels=8, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.reshape(x.shape[0], x.shape[1])
        print("avg_pool's size: ", y.size())
        print("avg_pool: ", y)

        y = self.fc(y)
        y = y.reshape(x.shape[0], x.shape[1], 1)
        print("Squeeze and Excitation's size: ", y.size())
        print("Squeeze and Excitation: ", y)

        return x * y


if __name__ == '__main__':
    x = torch.rand((12, 8, 100), dtype=torch.float)
    print("input's size: ", x.size())
    # print("input: ", x)
    model = SELayer(in_channels=8,reduction=4)
    y = model(x)
    print("output's size: ", y.size())
    # print("output: ", y)
