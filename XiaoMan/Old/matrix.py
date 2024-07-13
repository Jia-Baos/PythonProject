import torch
import torch.nn as nn

y = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0]
], dtype=torch.double)

y_hat = torch.tensor([
    [0.8, 0.2],
    [0.8, 0.2],
    [0.8, 0.2]
], dtype=torch.double)

W3 = torch.tensor([
    [0.8, 0.5],
    [0.8, 0.1],
    [0.6, 0.1],
], dtype=torch.double)

Z2 = torch.tensor([
    [0.76, 0.79, 0.82],
    [0.76, 0.79, 0.82],
    [0.76, 0.79, 0.82]
], dtype=torch.double)

S2 = torch.tensor([
    [1.14, 1.35, 1.51],
    [1.14, 1.35, 1.51],
    [1.14, 1.35, 1.51]
], dtype=torch.double)

W2 = torch.tensor([
    [0.6, 0.9, 0.1],
    [0.1, 0.0, 0.8],
    [0.8, 0.9, 1.0],
], dtype=torch.double)

Z1 = torch.tensor([
    [0.7, 0.8, 0.8],
    [0.7, 0.8, 0.8],
    [0.7, 0.8, 0.8]
], dtype=torch.double)

S1 = torch.tensor([
    [1.0, 1.1, 1.2],
    [0.8, 1.7, 1.5],
    [0.9, 1.5, 1.1]
], dtype=torch.double)

X = torch.tensor([
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 1]
], dtype=torch.double)

if __name__ == '__main__':

    # Z2 @ W3 = O, softmax(O) = y_hat
    g3 = torch.mm(Z2.transpose(0, 1), y_hat - y)
    print(g3)

    # Z1 @ W2 = S2, sigmoid(S2) = Z2
    sigmoid = nn.Sigmoid()
    g2_1 = torch.mm(y_hat - y, W3.transpose(0, 1))
    g2_2 = torch.mul(sigmoid(S2), torch.ones_like(S2) - sigmoid(S2))
    g2_3 = torch.mul(g2_1, g2_2)
    g2 = torch.mm(Z1.transpose(0, 1), g2_3)
    print(g2)

    # X1 @ W1 = S1, sigmoid(S1) = Z1
    g1_1 = torch.mm(g2_3, W2.transpose(0, 1))
    g1_2 = torch.mul(sigmoid(S1), torch.ones_like(S1) - sigmoid(S1))
    g1_3 = torch.mul(g1_1, g2_2)
    g1 = torch.mm(X.transpose(0, 1), g1_3)
    print(g1)
