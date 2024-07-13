import torch

bn_input = torch.rand((2, 3, 2, 2))

ln_input = torch.rand((2, 3, 4))


def batch_norm(x: torch.Tensor, eps=1e-05) -> torch.Tensor:
    alpha = torch.nn.Parameter(torch.ones(x.shape[1]))
    print("alpha size: \n", alpha.shape)
    # print("alpha: \n", alpha)
    beta = torch.nn.Parameter(torch.zeros(x.shape[1]))
    print("beta size: \n", beta.shape)
    # print("beta: \n", beta)
    mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
    print("mean size: \n", mean.shape)
    # print("mean: \n", mean)
    var = torch.var(x, dim=(0, 2, 3), keepdim=True, unbiased=False)
    print("var size: \n", var.shape)
    # print("var: \n", var)
    output = (x - mean) / torch.sqrt(var + eps)
    print("output size: \n", output.shape)
    output = output * alpha.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)
    print("alpha.view size: \n", alpha.view(1, -1, 1, 1).shape)
    print("beta.view size: \n", beta.view(1, -1, 1, 1).shape)
    return output


def batch_norm(x: torch.Tensor, eps=1e-05) -> torch.Tensor:
    alpha = torch.nn.Parameter(torch.ones(x.shape[1]))
    print("alpha size: \n", alpha.shape)
    # print("alpha: \n", alpha)
    beta = torch.nn.Parameter(torch.zeros(x.shape[1]))
    print("beta size: \n", beta.shape)
    # print("beta: \n", beta)
    mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
    print("mean size: \n", mean.shape)
    # print("mean: \n", mean)
    var = torch.var(x, dim=(0, 2, 3), keepdim=True, unbiased=False)
    print("var size: \n", var.shape)
    # print("var: \n", var)
    output = (x - mean) / torch.sqrt(var + eps)
    print("output size: \n", output.shape)
    output = output * alpha.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)
    print("alpha.view size: \n", alpha.view(1, -1, 1, 1).shape)
    print("beta.view size: \n", beta.view(1, -1, 1, 1).shape)
    return output


if __name__ == "__main__":
    bn = torch.nn.BatchNorm2d(3)
    print("BN input: \n", bn_input)
    print("Our BN output: \n", batch_norm(bn_input))
    print("Torch BN output: \n", bn(bn_input))
