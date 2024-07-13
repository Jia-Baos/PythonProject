import torch
from MyNet import XiaoManNet

model = XiaoManNet()
model.load_state_dict(torch.load('./best_model.pt'))

classes = ["good", "bad"]

if __name__ == '__main__':
    input = torch.tensor([
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 0, 0, 1]
    ], dtype=torch.float)
    with torch.no_grad():
        output = model(input)
        print(output)
        # 0代表好评，1代表差评
        predict = torch.argmax(output, dim=1)
        print(predict)
