import torch
import torch.nn as nn
from MyNet import XiaoManNet

x = torch.tensor([
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 1]
], dtype=torch.float)

label = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
], dtype=torch.float)

model = XiaoManNet()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

if __name__ == '__main__':
    model.train()
    for epoch in range(500):
        best_loss = 1
        y = model(x)
        current_loss = loss(y, label)
        print("predict value")
        print(y)
        print("current_loss: ")
        print(current_loss)

        optimizer.zero_grad()
        current_loss.backward()

        print("{}weight and bias".format(epoch))
        print(model.c3.weight)
        print(model.c2.weight)
        print(model.c1.weight)

        print("{}grad of weight and bias".format(epoch))
        print(model.c3.weight.grad)
        print(model.c2.weight.grad)
        print(model.c1.weight.grad)

        optimizer.step()

        print("{}weight and bias has updated".format(epoch))
        print(model.c3.weight)
        print(model.c2.weight)
        print(model.c1.weight)

        if current_loss < best_loss:
            torch.save(model.state_dict(), 'best_model.pt')
