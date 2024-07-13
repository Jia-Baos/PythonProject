import torch

W1 = torch.tensor([
    [0.5, 0.7, 0.6],
    [0.5, 0.4, 0.6],
    [0.4, 0.9, 1.0],
    [0.4, 0.8, 0.5]
], dtype=torch.float)

W1_bias = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float)

W2 = torch.tensor([
    [0.6, 0.9, 0.1],
    [0.1, 0.0, 0.8],
    [0.8, 0.9, 1.0],
], dtype=torch.float)

W2_bias = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float)

W3 = torch.tensor([
    [0.8, 0.5],
    [0.8, 0.1],
    [0.6, 0.1],
], dtype=torch.float)

W3_bias = torch.tensor([0.0, 0.0], dtype=torch.float)
