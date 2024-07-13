import torch
import torch.nn as nn

loss = nn.CrossEntropyLoss()

# X：
# 四维张量[N, C, H, W]
# N：每个batch图片的数量
# C：通道数（channels），即in_channels
# H：图片的高
# W：图片的宽
X = torch.tensor([[[
    [0.4, 0.7, 0.0],
    [0.3, 0.1, 0.1],
    [0.2, 0.3, 0.4],
    [0.5, 0.4, 0.7]
]]], dtype=torch.float)

W1 = torch.tensor([[[
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0]
]]])
W1_copy = torch.ones_like(W1)

# 注意区分[-0.1149, -0.8085]和[[-0.1149, -0.8085]]
# [[-0.1149, -0.8085]]，二维数组，size为(1, 2)
# [-0.1149, -0.8085]，一维数组，size为2
W2 = torch.tensor([[-0.1149, -0.8085]], dtype=torch.float)
W2_copy = torch.ones_like(W2)

y = torch.tensor([[1.0, 0.0]], dtype=torch.float)
curr_loss = 1

for epoch in range(100):
    print("the epoch: {}".format(epoch))

    # forward: step1
    Conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 3), stride=1, padding=0, bias=False)
    Conv1.weight.data = W1
    S1 = Conv1(X)
    print("size of S1: {}".format(S1.size()))
    print(S1)

    # forward: step2
    # ReLU的输入为二维
    S1 = S1.view(3, -1)
    Relu = nn.ReLU()
    P1 = Relu(S1)
    print("size of P1: {}".format(P1.size()))
    print(P1)

    # forward: step3
    # MaxPool1D输入的前两个维度为batch, channels
    P1 = P1.permute(1, 0)
    Maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=0)
    Z = Maxpool(P1)
    print("size of Z: {}".format(Z.size()))
    print(Z)

    # forward: step4
    # 全连接层计算公式为: X @ W.transpose(0, 1) + b
    linear = nn.Linear(in_features=1, out_features=2, bias=False)
    linear.weight = torch.nn.Parameter(W2.transpose(0, 1))
    O = linear(Z)
    print("size of O: {}".format(O.size()))
    print(O)

    # forward: step5
    # Softmax的输入为二维
    softmax = nn.Softmax(dim=1)
    y_hat = softmax(O)
    print("size of y_hat: {}".format(y_hat.size()))
    print(y_hat)
    print(loss(y_hat, y))
    if loss(y_hat, y) < curr_loss:
        curr_loss = loss(y_hat, y)
        W1_copy = W1.clone()
        W2_copy = W2.clone()

    # backward: step1
    g2_1 = y_hat - y
    print("size of g2_1: {}".format(g2_1.size()))
    print(g2_1)
    g2 = torch.mm(Z.transpose(0, 1), g2_1)
    print("size of g2: {}".format(g2.size()))
    print(g2)

    # backward: step2
    g1_1 = torch.mm(g2_1, W2.transpose(0, 1))
    print("size of g1_1: {}".format(g1_1.size()))
    print(g1_1)

    g1_2 = torch.zeros_like(P1)
    index = torch.argmax(P1, dim=1)
    # 此处为二维数组切片
    g1_2[:, index] = 1.0
    g1_3 = torch.mm(g1_1, g1_2)
    print("size of g1_3: {}".format(g1_3.size()))
    print(g1_3)

    # 此处g1_3 = g1_4，因为值均大于0
    g1_4 = torch.zeros_like(g1_3)
    size = g1_4.size(1)
    for i in range(size):
        if S1[i, :] > 0:
            g1_4[:, i] = g1_3[:, i]
    print("size of g1_4: {}".format(g1_4.size()))
    print(g1_4)

    # backward: step5
    g1_5 = (g1_4.unsqueeze(0)).unsqueeze(0)
    g1_5 = g1_5.permute(0, 1, 3, 2)
    print("size of g1_5: {}".format(g1_5.size()))
    print(g1_5)
    Conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 1), stride=1, padding=0, bias=False)
    Conv2.weight.data = g1_5.clone().detach()
    g1 = Conv2(X)
    print("size of g1: {}".format(g1.size()))
    print(g1)

    W1 = W1 - torch.mul(g1, 0.1)
    W2 = W2 - torch.mul(g2, 0.1)

# 训练后得到的权重
print("W1_copy: ")
print(W1_copy)
print("W2_copy: ")
print(W2_copy)
