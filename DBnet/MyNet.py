import torch
import torch.nn as nn
import torch.nn.functional as func


# Model
class ResNet50BasicBlock(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super(ResNet50BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=mid_c, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.conv2 = nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_c)
        self.conv3 = nn.Conv2d(in_channels=mid_c, out_channels=out_c, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = x + out
        out = self.relu(out)
        return out


class ResNet50DownBlock(nn.Module):
    def __init__(self, in_c, mid_c, out_c, stride=1):
        super(ResNet50DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=mid_c, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.conv2 = nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_c)
        self.conv3 = nn.Conv2d(in_channels=mid_c, out_channels=out_c, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.conv1_1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, stride=stride, kernel_size=1)
        self.bn1_1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1_1(x)
        x1 = self.bn1_1(x1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = x1 + out
        out = self.relu(out)
        return out


class DBHead(nn.Module):
    def __init__(self, in_channels, k=50):
        super().__init__()
        self.k = k
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 2, 2),
            nn.Sigmoid())

        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 2, 2),
            nn.Sigmoid())

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x):
        probability_maps = self.binarize(x)
        threshold_maps = self.thresh(x)

        approximate_binary_maps = self.step_function(probability_maps, threshold_maps)
        return probability_maps, threshold_maps, approximate_binary_maps


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 因为原论文中是下采样5次，所以这里对代码进行修改，使之满足要求
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.bottleneck1_1 = ResNet50DownBlock(in_c=64, mid_c=64, out_c=256, stride=2)

        # self.bottleneck1_1 = ResNet50DownBlock(in_c=64, mid_c=64, out_c=256)
        self.bottleneck1_2 = ResNet50BasicBlock(in_c=256, mid_c=64, out_c=256)
        self.bottleneck1_3 = ResNet50BasicBlock(in_c=256, mid_c=64, out_c=256)

        self.bottleneck2_1 = ResNet50DownBlock(in_c=256, mid_c=128, out_c=512, stride=2)
        self.bottleneck2_2 = ResNet50BasicBlock(in_c=512, mid_c=128, out_c=512)
        self.bottleneck2_3 = ResNet50BasicBlock(in_c=512, mid_c=128, out_c=512)
        self.bottleneck2_4 = ResNet50BasicBlock(in_c=512, mid_c=128, out_c=512)

        self.bottleneck3_1 = ResNet50DownBlock(in_c=512, mid_c=256, out_c=1024, stride=2)
        self.bottleneck3_2 = ResNet50BasicBlock(in_c=1024, mid_c=256, out_c=1024)
        self.bottleneck3_3 = ResNet50BasicBlock(in_c=1024, mid_c=256, out_c=1024)
        self.bottleneck3_4 = ResNet50BasicBlock(in_c=1024, mid_c=256, out_c=1024)
        self.bottleneck3_5 = ResNet50BasicBlock(in_c=1024, mid_c=256, out_c=1024)
        self.bottleneck3_6 = ResNet50BasicBlock(in_c=1024, mid_c=256, out_c=1024)

        self.bottleneck4_1 = ResNet50DownBlock(in_c=1024, mid_c=512, out_c=2048, stride=2)
        self.bottleneck4_2 = ResNet50BasicBlock(in_c=2048, mid_c=512, out_c=2048)
        self.bottleneck4_3 = ResNet50BasicBlock(in_c=2048, mid_c=512, out_c=2048)

        # FPN
        self.conv_c5_m5 = nn.Conv2d(in_channels=2048, out_channels=64, kernel_size=1)
        self.conv_m5_p5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_c4_m4 = nn.Conv2d(in_channels=1024, out_channels=64, kernel_size=1)
        self.conv_m4_p4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_c3_m3 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1)
        self.conv_m3_p3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_c2_m2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)
        self.conv_m2_p2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # CON-CAT's conv
        self.conv_p2_p2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_p3_p3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_p4_p4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_p5_p5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # Db_head
        self.head = DBHead(in_channels=256)

    def forward(self, x):
        # resnet预处理阶段
        x = self.conv1(x)
        x = self.bn(x)
        c1 = self.relu(x)
        x = self.pool1(c1)
        print("c1", x.size())

        # 进入第一个block
        x = self.bottleneck1_1(x)
        x = self.bottleneck1_2(x)
        c2 = self.bottleneck1_3(x)
        print("c2", c2.size())

        # 进入第二个block
        x = self.bottleneck2_1(c2)
        x = self.bottleneck2_2(x)
        x = self.bottleneck2_3(x)
        c3 = self.bottleneck2_4(x)
        print("c3", c3.size())

        # 进入第三个block
        x = self.bottleneck3_1(c3)
        x = self.bottleneck3_2(x)
        x = self.bottleneck3_3(x)
        x = self.bottleneck3_4(x)
        x = self.bottleneck3_5(x)
        c4 = self.bottleneck3_6(x)
        print("c4", c4.size())

        # 进入第四个block
        x = self.bottleneck4_1(c4)
        x = self.bottleneck4_2(x)
        c5 = self.bottleneck4_3(x)
        print("c5", c5.size())

        # 特征图第一层
        m5 = self.conv_c5_m5(c5)
        p5 = self.conv_m5_p5(m5)  # 这个要做什么用，平滑后输入CON-CAT
        print("m5", m5.size())

        c4_m4 = self.conv_c4_m4(c4)
        m5_x2 = func.interpolate(m5, (m5.shape[-2] * 2, m5.shape[-1] * 2), mode='bilinear', align_corners=True)
        m4 = c4_m4 + m5_x2
        p4 = self.conv_m4_p4(m4)  # 这个要做什么用，平滑后输入CON-CAT
        print("m4", m4.size())

        # 特征图第二层
        c3_m3 = self.conv_c3_m3(c3)
        m4_x2 = func.interpolate(m4, (m4.shape[-2] * 2, m4.shape[-1] * 2), mode='bilinear', align_corners=True)
        m3 = c3_m3 + m4_x2
        p3 = self.conv_m3_p3(m3)  # 这个要做什么用，平滑后输入CON-CAT
        print("m3", m3.size())

        # 特征图第三层
        c2_m2 = self.conv_c2_m2(c2)
        m3_x2 = func.interpolate(m3, (m3.shape[-2] * 2, m3.shape[-1] * 2), mode='bilinear', align_corners=True)
        m2 = c2_m2 + m3_x2
        p2 = self.conv_m2_p2(m2)  # 补上之前没有做的平滑
        print("m2", m2.size())

        # 开始制备CON-CAT的输入
        p2 = self.conv_p2_p2(p2)  # CON-CAT输入前的卷积

        p3 = self.conv_p3_p3(p3)  # CON-CAT输入前的卷积
        p3 = func.interpolate(p3, (p3.shape[-2] * 2, p3.shape[-1] * 2), mode='bilinear', align_corners=True)

        p4 = self.conv_p4_p4(p4)  # CON-CAT输入前的卷积
        p4 = func.interpolate(p4, (p4.shape[-2] * 4, p4.shape[-1] * 4), mode='bilinear', align_corners=True)

        p5 = self.conv_p5_p5(p5)  # CON-CAT输入前的卷积
        p5 = nn.functional.interpolate(p5, (p5.shape[-2] * 8, p5.shape[-1] * 8), mode='bilinear', align_corners=True)

        feature = torch.cat((p2, p3, p4, p5), dim=1)
        print("feature's size: ", feature.size())

        probability_maps, threshold_maps, approximate_binary_maps = self.head(feature)

        return probability_maps, threshold_maps, approximate_binary_maps


if __name__ == '__main__':
    # 测试计划，直接利用224*224的图像测试，使得网络过拟合，查看是否可以准确找到边缘，之后再跑各种数据集
    x = torch.rand((4, 3, 224, 224), dtype=torch.float)
    model = MyModel()
    real_probability_maps, real_threshold_maps, real_approximate_binary_maps = model(x)
    print(real_probability_maps.size())
    print(real_threshold_maps.size())
    print(real_approximate_binary_maps.size())
