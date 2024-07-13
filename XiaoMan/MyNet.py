import torch
import torch.nn as nn
from Myutils import W1, W2, W3


# define the struct of net
class XiaoManNet(nn.Module):
    # initialization of net
    def __init__(self):
        super(XiaoManNet, self).__init__()

        self.c1 = nn.Linear(in_features=4, out_features=3, bias=False)
        self.c1.weight = torch.nn.Parameter(W1.transpose(0, 1))
        # self.c1.bias = torch.nn.Parameter(W1_bias)
        self.s1 = nn.Sigmoid()
        # print(self.c1.weight)
        # print(self.c1.bias)

        self.c2 = nn.Linear(in_features=3, out_features=3, bias=False)
        self.c2.weight = torch.nn.Parameter(W2.transpose(0, 1))
        # self.c2.bias = torch.nn.Parameter(W2_bias)
        self.s2 = nn.Sigmoid()
        # print(self.c2.weight)
        # print(self.c2.bias)

        self.c3 = nn.Linear(in_features=3, out_features=2, bias=False)
        self.c3.weight = torch.nn.Parameter(W3.transpose(0, 1))
        # self.c3.bias = torch.nn.Parameter(W3_bias)
        self.s3 = nn.Softmax(dim=1)
        # print(self.c3.weight)
        # print(self.c3.bias)

    def forward(self, X):
        S1 = self.c1(X)
        Z1 = self.s1(S1)
        S2 = self.c2(Z1)
        Z2 = self.s2(S2)
        S3 = self.c3(Z2)
        Z3 = self.s3(S3)
        return Z3


if __name__ == "__main__":
    # test XiaoManNet
    x = torch.tensor([
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1]
    ], dtype=torch.float)
    model = XiaoManNet()
    y = model(x)
    print(y)
