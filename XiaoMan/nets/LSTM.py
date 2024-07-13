import torch
import torch.nn as nn


class MyLSTM(nn.Module):
    def __init__(self, in_features=6, cell_state_size=2, out_features=2):
        """

        :param in_features: 输入x的长度，即词向量的长度
        :param cell_state: cell_state的长度，即隐藏状态的长度
        :param out_features: 输出h的长度
        """
        super(MyLSTM, self).__init__()
        self.in_features = in_features
        self.cell_state_size = cell_state_size
        self.out_features = out_features

        self.forget_gate = nn.Linear(self.in_features + self.cell_state_size, self.cell_state_size)

        self.input_gate1 = nn.Linear(self.in_features + self.cell_state_size, self.cell_state_size)
        self.input_gate2 = nn.Linear(self.in_features + self.cell_state_size, self.cell_state_size)

        self.output_gate = nn.Linear(self.in_features + self.cell_state_size, self.cell_state_size)

        self.linear_cls = nn.Linear(self.cell_state_size, out_features)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # x: [batch, seq_len, in_features] -> [seq_len, batch, in_features]
        x = x.permute(1, 0, 2)
        cell_state = torch.zeros((1, x.shape[1], self.cell_state_size), dtype=torch.float)
        output_sequence = torch.zeros((x.shape[0], x.shape[1], self.out_features), dtype=torch.float)

        for i in range(x.shape[0]):
            print("*********************************************************")

            # 各门的输入
            input = torch.cat((cell_state, torch.unsqueeze(x[i, :, :], dim=0)), dim=2)
            print("original cell_state: ", cell_state.size())

            # 遗忘门
            forget_gate_output = self.forget_gate(input)
            forget_gate_output = self.sigmoid(forget_gate_output)

            # 更新cell_state
            cell_state = torch.mul(cell_state, forget_gate_output)
            print("cell_state first update: ", cell_state.size())

            # 输入门
            input_gate_output1 = self.input_gate1(input)
            input_gate_output1 = self.sigmoid(input_gate_output1)
            input_gate_output2 = self.input_gate2(input)
            input_gate_output2 = self.tanh(input_gate_output2)
            input_gate_output = torch.mul(input_gate_output1, input_gate_output2)

            # 更新cell_state
            cell_state = torch.add(cell_state, input_gate_output)
            print("cell_state second update: ", cell_state.size())

            # 输出门
            output_gate_output1 = self.output_gate(input)
            output_gate_output1 = self.sigmoid(output_gate_output1)
            output_gate_output2 = self.tanh(cell_state)
            output_gate_output = torch.mul(output_gate_output1, output_gate_output2)
            print("output_gate_output: ", output_gate_output.size())

            # 输出门在经过一个全连接层得到当前帧的预测
            output = self.linear_cls(output_gate_output)
            print("output's size: ", output.size())
            output_sequence[i, :, :] = output

        output_sequence = output_sequence.permute(1, 0, 2)
        return output_sequence, cell_state


if __name__ == '__main__':
    """
    此处错误可参考连接： https://blog.csdn.net/Mr_wuliboy/article/details/90520011
    关于这个错误仔细思考了以下，原因在于forward的返回结果有两个————output、hidden state
    所以要用 output, hidden_state = model(x) 的形式来接受结果
    如果直接 y = model(x) 则会报错，因为此时返回的是一个元组
    哈哈哈哈哈，很有意思的一件事哈
    """

    # x: [batch, seq_len, in_features]
    x = torch.rand((16, 128, 128), dtype=torch.float)
    print("input's size: ", x.size())
    model = MyLSTM(in_features=x.shape[2], cell_state_size=4, out_features=2)
    output, cell_state = model(x)
    print(output)
    print(type(output))
    print(cell_state)
    print(type(cell_state))
