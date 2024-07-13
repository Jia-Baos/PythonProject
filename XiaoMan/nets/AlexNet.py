import torch
import torch.nn as nn


class LRN(nn.Module):
    def __init__(self, in_channels: int, k=2, n=5, alpha=1.0e-4, beta=0.75):
        # 把所有用的到的参数进行赋值，参数名和论文里面是基本一致的
        super(LRN, self).__init__()
        # 特征图的通道数，就是论文里的N
        self.in_channels = in_channels
        self.k = k
        self.n = n
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        tmp = x.pow(2)
        div = torch.zeros(tmp.size())

        for batch in range(tmp.size(0)):
            for channel in range(tmp.size(1)):
                st = max(0, channel - self.n // 2)
                ed = min(channel + self.n // 2, tmp.size(1) - 1) + 1
                div[batch, channel] = tmp[batch, st:ed].sum(dim=0)  # 切片操作

        out = x / (self.k + self.alpha * div).pow(self.beta)
        return out


class AlexNet(nn.Module):
    def __init__(self, nclass=10):
        super(AlexNet, self).__init__()
        self.nclass = nclass
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout(p=0.5, inplace=True)

        self.C1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.C1.bias.data = torch.zeros(self.C1.bias.data.size())
        self.N1 = LRN(96)

        self.C2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.C2.bias.data = torch.ones(self.C2.bias.data.size())
        self.N2 = LRN(256)

        self.C3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.C3.bias.data = torch.zeros(self.C3.bias.data.size())

        self.C4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.C4.bias.data = torch.ones(self.C4.bias.data.size())

        self.C5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.C5.bias.data = torch.ones(self.C5.bias.data.size())

        self.F1 = nn.Linear(256 * 6 * 6, 4096)
        self.F2 = nn.Linear(4096, 4096)
        self.F3 = nn.Linear(4096, nclass)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data = torch.normal(torch.zeros(m.weight.data.size()), torch.ones(m.weight.data.size()) * 0.01)
                if isinstance(m, nn.Linear):
                    m.bias.data = torch.ones(m.bias.data.size())

    def forward(self, x):
        x = self.pool(self.N1(self.act(self.C1(x))))
        print("first conv, act, lrn, pool: ", x.size())

        x = self.pool(self.N2(self.act(self.C2(x))))
        print("second conv, act, lrn, pool: ", x.size())

        x = self.act(self.C3(x))
        print("third conv, act: ", x.size())

        x = self.act(self.C4(x))
        print("fourth conv, act: ", x.size())

        x = self.pool(self.act(self.C5(x)))
        print("fifth conc, act, pool", x.size())

        # x = x.view(-1, 256 * 6 * 6)
        x = torch.flatten(x, 1)
        print("ready to fcn: ", x.size())

        x = self.dropout(self.act(self.F1(x)))
        print("first fcn, act, dropout: ", x.size())

        x = self.dropout(self.act(self.F2(x)))
        print("second fcn, act, dropout: ", x.size())

        x = self.act(self.F3(x))
        print("last act: ", x.size())
        return x


if __name__ == '__main__':
    x = torch.rand((4, 3, 227, 227), dtype=torch.float)
    print("input's size: ", x.size())
    model = AlexNet()
    y = model(x)
    print("output's size: ", y.size())
