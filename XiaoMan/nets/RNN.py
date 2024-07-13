import torch
import torch.nn as nn


class MyRNN(nn.Module):
    def __init__(self, in_features=6, hidden_size=4, out_features=2):
        super(MyRNN, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_features = out_features
        self.linear1 = nn.Linear(self.in_features + self.hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.out_features)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # x: [batch, seq_len, in_features] -> [seq_len, batch, in_features]
        input = x.permute(1, 0, 2)
        # 初始化hidden_state
        pre_hidden_state = torch.zeros((1, input.shape[1], self.hidden_size), dtype=torch.float)
        # 初始化RNN输出序列
        output = torch.zeros((input.shape[0], input.shape[1], self.out_features), dtype=torch.float)

        for i in range(input.shape[0]):
            print("num of seq_len: ", i)
            print("input_pre's size: ", torch.unsqueeze(input[i, :, :], dim=0).size())
            print("pre_hidden_state's size: ", pre_hidden_state.size())
            input_hidden_cat = torch.cat((pre_hidden_state, torch.unsqueeze(input[i, :, :], dim=0)), dim=2)
            print("input_hidden_cat's size: ", input_hidden_cat.size())
            curr_hidden_state = self.linear1(input_hidden_cat)
            curr_hidden_state = self.tanh(curr_hidden_state)
            print("hidden_state's size: ", curr_hidden_state.size())

            # 修正下一轮所使用的hidden_state
            pre_hidden_state = curr_hidden_state
            output_temp = self.linear2(curr_hidden_state)
            print("output_temp's size: ", output_temp.size())
            output[i, :, :] = output_temp

        output = output.permute(1, 0, 2)
        return output, curr_hidden_state


if __name__ == '__main__':
    """
    此处错误可参考连接： https://blog.csdn.net/Mr_wuliboy/article/details/90520011
    关于这个错误仔细思考了以下，原因在于forward的返回结果有两个————output、hidden state
    所以要用 output, hidden_state = model(x) 的形式来接受结果
    如果直接 y = model(x) 则会报错，因为此时返回的是一个元组
    哈哈哈哈哈，很有意思的一件事哈
    """

    # x: [batch, seq_len, in_features]
    x = torch.rand((4, 10, 6), dtype=torch.float)
    print("input's size: ", x.size())
    print("*********************************************")
    model = MyRNN(in_features=x.shape[2], hidden_size=4, out_features=2)
    output, hidden_state = model(x)
    print(output)
    print(type(output))
    print(hidden_state)
    print(type(hidden_state))
