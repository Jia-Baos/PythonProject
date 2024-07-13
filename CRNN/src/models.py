# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/27 16:45
# @Author  : Jis-Baos
# @File    : Mynet.py

import torch
import torch.nn as nn
from dict_tools import NUM_CLASSES


# LSTM类
class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        # *2因为使用双向LSTM，两个方向隐层单元拼在一起
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN_Better(nn.Module):
    def __init__(self, imgH, nc, nclass, nh=256, n_rnn=2, leakyRelu=False):
        """
        :param imgH: 图片高度
        :param nc: 输入图片通道数
        :param nclass: 分类数目
        :param nh: rnn隐藏层神经元节点数
        :param n_rnn: rnn的层数
        :param leakyRelu: 是否使用LeakyRelu
        """
        super(CRNN_Better, self).__init__()
        assert imgH % 16 == 0, "imgH has to be a multiple of 16 图片高度必须为16的倍数"

        ks = [3, 3, 3, 3, 3, 3, 2]  # 卷积层卷积尺寸3表示3x3，2表示2x2
        ps = [1, 1, 1, 1, 1, 1, 0]  # padding大小
        ss = [1, 1, 1, 1, 1, 1, 1]  # stride大小
        nm = [64, 128, 256, 256, 512, 512, 512]  # 卷积核个数

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]  # 确定输入channel维度
            nOut = nm[i]  # 确定输出channel维度

            cnn.add_module(
                "conv{0}".format(i), nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i])
            )  # 添加卷积层
            # BN层
            if batchNormalization:
                cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(nOut))
            # Relu激活层
            if leakyRelu:
                cnn.add_module("relu{0}".format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module("relu{0}".format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module("pooling{0}".format(0), nn.MaxPool2d(2, 2))

        convRelu(1)
        cnn.add_module("pooling{0}".format(1), nn.MaxPool2d(2, 2))

        convRelu(2, True)

        convRelu(3)
        cnn.add_module("pooling{0}".format(2), nn.MaxPool2d((2, 1), (2, 1)))

        convRelu(4, True)

        convRelu(5)
        cnn.add_module("pooling{0}".format(3), nn.MaxPool2d((2, 1), (2, 1)))

        convRelu(6, True)

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh), BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        # print("CNN's output: ", conv.size())

        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(dim=-2)
        # conv = conv.permute(2, 0, 1)  # [w, b, c] [24, b, 512]
        conv = conv.permute(0, 2, 1)
        # print("RNN's input: ", conv.size())

        # rnn features
        output = self.rnn(conv)

        return output


class CRNN(nn.Module):
    def __init__(self, num_classes):
        """
        网络的输入为：W*32的灰度图
        """
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(3, 3), stride=1, padding=1
        )
        self.activation1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1
        )
        self.activation2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1
        )
        self.batch_norm3 = nn.BatchNorm2d(num_features=256)


        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1
        )
        self.activation4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv5 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1
        )
        self.batch_norm5 = nn.BatchNorm2d(num_features=512)
        
        self.conv6 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1
        )
        self.activation6 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv7 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=(2, 2), stride=1, padding=0
        )
        self.batch_norm7 = nn.BatchNorm2d(num_features=512)

        # input_size: embedding_dim
        # hidden_size: hidden_layer's size
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
        )

        self.transcript = nn.Linear(512, num_classes)

    def forward(self, x):
        """

        :param x: [B, 1, W, 32]
        :return: [B, W,num_classes]
        """
        # CNN
        output = self.conv1(x)
        output = self.activation1(output)
        # print(output.size())

        output = self.conv2(output)
        output = self.activation2(output)
        # print(output.size())

        output = self.conv3(output)
        output = self.batch_norm3(output)
        # print(output.size())

        output = self.conv4(output)
        output = self.activation4(output)
        # print(output.size())

        output = self.conv5(output)
        output = self.batch_norm5(output)
        # print(output.size())

        output = self.conv6(output)
        output = self.activation6(output)
        # print(output.size())

        output = self.conv7(output)
        output = self.batch_norm7(output)
        # print("CNN's output: ", output.size())

        # RNN
        output = torch.squeeze(output, dim=-2)
        # batch, embedding_dim, seq_len -> batch, seq_len, embedding_dim

        output = output.permute(0, 2, 1)
        # print("RNN's input: ", output.size())

        output, (h1, h2) = self.rnn(output)
        # print("RNN's output: ", output.size())
        # print(output[0][0])

        h1 = h1.permute(1, 0, 2)
        h2 = h2.permute(1, 0, 2)
        # print("RNN's h1: ", h1.size())
        # print("RNN's h2: ", h2.size())

        # Transcript
        output = self.transcript(output)
        # print("Transcript's output: ", output.size())

        return output


if __name__ == "__main__":
    x = torch.rand(size=(4, 1, 32, 100), dtype=torch.float)

    crnn = CRNN(num_classes=NUM_CLASSES)
    # Batch * size * Input sequence length for CTC * Number of classes (including blank)
    output = crnn(x)
    print("crnn: {}".format(output.size()))

    crnn_better = CRNN_Better(32, 1, NUM_CLASSES, 256)
    output = crnn_better(x)
    print("crnn_better: {}".format(output.size()))
