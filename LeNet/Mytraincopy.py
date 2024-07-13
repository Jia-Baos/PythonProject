import torch
from torch import nn
from MyLeNet import MyNet
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time
import datetime

checkpoints_dir = "D:\\PythonProject\\LeNet\\checkpoints"

# 将数据转化为tensor
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)

# 加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True)

# 如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 调用MyNet模型，将模型数据转到GPU
model = MyNet().to(device)
model.load_state_dict(torch.load("D:\\PythonProject\\LeNet\\checkpoints\\best_model.pt"))

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()

# 定义一个优化器
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# 学习率每隔十轮，变为原来的0.1
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (x, y) in enumerate(dataloader):
        # 前向传播
        x, y = x.to(device), y.to(device)
        output = model(x)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, dim=1)

        cur_acc = torch.sum(y == pred) / output.shape[0]

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1

    print("train loss: " + str(loss / n))
    print("train acc: " + str(current / n))

    log_file = open(os.path.join(checkpoints_dir, "log.txt"), "a+")
    log_file.write("Epoch %d | loss = %.3f | current = %.3f\n" % (epoch, loss / n, current / n))
    log_file.flush()
    log_file.close()


def val(dataloader, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            # 前向传播
            x, y = x.to(device), y.to(device)
            output = model(x)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, dim=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]

            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1

        print("val loss: " + str(loss / n))
        print("val acc: " + str(current / n))

        log_file = open(os.path.join(checkpoints_dir, "log.txt"), "a+")
        log_file.write("Epoch %d | loss = %.3f | current = %.3f\n" % (epoch, loss / n, current / n))
        log_file.flush()
        log_file.close()

        return current / n


# 开始训练
if __name__ == '__main__':
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    epochs = 50
    min_acc = 0

    for epoch in range(epochs):
        print("\nepoch: %d" % (epoch + 1))

        with open(os.path.join(checkpoints_dir, "log.txt"), "a+") as log_file:
            # strftime()，格式化输出时间
            localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 打印训练时间
            log_file.write(localtime)
            log_file.write("\n======================training epoch %d======================\n" % (epoch + 1))

        t1 = time.time()
        train(train_dataloader, model, loss_fn, optimizer)
        t2 = time.time()

        print("Training consumes %.2f second" % (t2 - t1))
        with open(os.path.join(checkpoints_dir, "log.txt"), "a+") as log_file:
            log_file.write("Training consumes %.2f second\n" % (t2 - t1))

        with open(os.path.join(checkpoints_dir, "log.txt"), "a+") as log_file:
            log_file.write("\n======================validate epoch %d======================\n" % (epoch + 1))
        t1 = time.time()
        a = val(test_dataloader, model, loss_fn)
        t2 = time.time()

        print("Validation consumes %.2f second" % (t2 - t1))
        with open(os.path.join(checkpoints_dir, "log.txt"), "a+") as log_file:
            log_file.write("Validation consumes %.2f second\n\n" % (t2 - t1))

        # 保存最好的模型权重
        if a > min_acc:
            folder = 'checkpoints'
            if not os.path.exists(folder):
                os.mkdir('checkpoints')
            min_acc = a
            print("save best model\n")
            torch.save(model.state_dict(), 'checkpoints/best_model.pt')
    print("Done!!!")