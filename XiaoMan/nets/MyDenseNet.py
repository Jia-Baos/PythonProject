import math
import torch
import torch.nn as nn
import torch.nn.functional as func


class Bottleneck(nn.Module):
    """
    Block内的一个小单元
    channels -> 4 * growth_rate -> growth_rate
    """
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        y = self.conv1(func.relu(self.bn1(x)))
        y = self.conv2(func.relu(self.bn2(y)))
        # 在channels上进行叠加
        x = torch.cat([y, x], dim=1)
        return x


class Transition(nn.Module):
    """
    The transition is responsible for connection of two blocks
    """
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(func.relu(self.bn(x)))
        x = func.avg_pool2d(x, kernel_size=2)
        return x


class DenseNet(nn.Module):
    """
    block: which is equivalent to the Bottleneck
    num_block: which is responsible for the number of Bottleneck in Denseblock
    growth_rate: which is responsible for the increasement of channels
        Bottleneck: in_planes -> in_planes + growth_rate
        Denseblock: num_planes -> (num_planes + Denseblock's chennels * growth_rate) * reduction
    reduction: the ratio of downsample
    num_classes: the number of classes in result
    fn_size: the size of every channel before enter the fully connected layer
    pool_size: 
    """
    def __init__(self, block, num_block, growth_rate=12, reduction=0.5, num_classes=10, fn_size=1, pool_size=7):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.pool_size = pool_size

        # 进入DenseBlock前的预处理
        num_planes = 2 * growth_rate
        # channels = 3
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # channels = 24
        # the num_planes is equivalent to in_planes of Bottleneck
        self.dense1 = self._make_dense_layers(block, num_planes, num_block[0])
        num_planes += num_block[0] * growth_rate
        # channels = 24 + 6 * 12 = 96
        # the reduction is equivalent the stride of avg_pool2d in Transition block 
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        # channels = 96 / 2 = 48
        self.dense2 = self._make_dense_layers(block, num_planes, num_block[1])
        num_planes += num_block[1] * growth_rate
        # channels = 48 + 12 * 12 = 192
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        # channels = 192 / 2 = 96
        self.dense3 = self._make_dense_layers(block, num_planes, num_block[2])
        num_planes += num_block[2] * growth_rate
        # channels = 96 + 24 * 12 = 384
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        # channels = 384 / 2 = 192
        self.dense4 = self._make_dense_layers(block, num_planes, num_block[3])
        num_planes += num_block[3] * growth_rate
        # channels = 192 + 16 * 12 = 384
        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes * fn_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense_layers(self, block, in_planes, num_block):
        """
        in fact, the block in this func is the Bottleneck 
        """
        layers = []
        for i in range(num_block):
            layers.append(block(in_planes, self.growth_rate))
            # the result'channels of Bottleneck's forward is in_planes +grow_rate
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        print("input", x.size())
        x = self.conv1(x)
        print("pre_conv: 7x7 S2", x.size())
        x = self.pool1(x)
        print("pre_pool: 3x3 S2", x.size())
        x = self.trans1(self.dense1(x))
        print("first_Denseblock: ", x.size())
        x = self.trans2(self.dense2(x))
        print("second_Denseblock: ", x.size())
        x = self.trans3(self.dense3(x))
        print("third_Denseblock: ", x.size())
        x = self.dense4(x)
        print("fourth_Denseblock without transition:", x.size())
        x = func.avg_pool2d(func.relu(self.bn(x)), self.pool_size)
        print("global average pool: 7x7 ", x.size())
        x = x.view(x.size(0), -1)
        print("flatten the result of GAP: ", x.size())
        x = self.linear(x)
        x = nn.Softmax(dim=1)(x)
        print("Softmax the output of linear layer: ", x.size())

        return x


def DenseNet121(fn_size=1, pool_size=7, num_classes=4):
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12,
                    num_classes=num_classes, fn_size=fn_size, pool_size=pool_size)


def DenseNet169(fn_size=1, pool_size=7, num_classes=4):
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32,
                    num_classes=num_classes, fn_size=fn_size, pool_size=pool_size)


def DenseNet201(fn_size=1, pool_size=7, num_classes=4):
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32,
                    num_classes=num_classes, fn_size=fn_size, pool_size=pool_size)


def DenseNet161(fn_size=1, pool_size=7, num_classes=4):
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48,
                    num_classes=num_classes, fn_size=fn_size, pool_size=pool_size)


if __name__ == "__main__":
    # size, fn_size, pool_size = 1024, 16, 8
    # size, fn_size, pool_size = 512, 4, 8
    # size, fn_size, pool_size = 256, 1, 8
    size, fn_size, pool_size = 256, 4, 4
    test_input = torch.rand(1, 3, size, size)
    model = DenseNet121(fn_size, pool_size, 10)
    # print(model)
    # summary(model, (3, size, size))
    output = model(test_input)
    print(output)
