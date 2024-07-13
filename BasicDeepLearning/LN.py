import torch

ln_input = torch.rand((2, 3, 4))


def layer_norm(x: torch.Tensor, eps=1e-05) -> torch.Tensor:
    alpha = torch.nn.Parameter(torch.ones(x.shape[-1]))
    print("alpha size: \n", alpha.shape)
    # print("alpha: \n", alpha)
    beta = torch.nn.Parameter(torch.zeros(x.shape[-1]))
    print("beta size: \n", beta.shape)
    # print("beta: \n", beta)
    mean = torch.mean(x, dim=-1, keepdim=True)
    print("mean size: \n", mean.shape)
    # print("mean: \n", mean)
    var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    print("var size: \n", var.shape)
    # print("var: \n", var)
    output = (x - mean) / torch.sqrt(var + eps)
    print("output size: \n", output.shape)
    output = output * alpha + beta
    return output


if __name__ == "__main__":
    ln = torch.nn.LayerNorm(4)
    print("BN input: \n", ln_input)
    print("Our BN output: \n", layer_norm(ln_input))
    print("Torch BN output: \n", ln(ln_input))
