# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/28 16:44
# @Author  : Jis-Baos
# @File    : Mytrain.py

# 此文件负责网络训练
import torch
from torch import nn
from src.models import CRNN
from src.dataset import DataSet
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import os
import time
import datetime

# 数据存放路径
data_dir = "D:\\PythonProject\\CRNN\\data_alpha"
# 权重存放路径
checkpoints_dir = "D:\\PythonProject\\CRNN\\checkpoints"

# 加载训练数据集
dataset = MyDataSet(data_dir, mode="train")
train_dataloader = DataLoader(
    dataset, batch_size=4, shuffle=False, num_workers=0, drop_last=False
)

# 加载验证数据集
dataset = MyDataSet(data_dir, mode="val")
val_dataloader = DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False
)

# 如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 调用MyCRNN模型，将模型数据转到GPU
model = MyCRNN(num_classes=53)
# model.load_state_dict(torch.load("D:\\PythonProject\\CRNN\\checkpoints\\best_model.pt"))

# 定义损失函数
loss_func = nn.CTCLoss(blank=0, reduction="mean")

# 定义一个优化器
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 学习率每隔十轮，变为原来的0.1
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


# 定义训练函数
def train(dataloader, model, loss_func, optimizer):
    model.train()
    train_avg_loss = 0.0
    for batch, (image, image_length, label, label_length) in enumerate(dataloader):
        print("batch: ", batch)
        # 前向传播
        image = torch.unsqueeze(image, dim=1)
        image, label = image.to(device), label.to(device)
        predict_label = model(image)
        # batch, seq_len(fixed), nun_classes -> seq_len(fixed), batch, nun_classes
        predict_label = predict_label.permute(1, 0, 2)
        # print("predict_label's size: ", predict_label.size())
        # print("label's size: ", label.size())
        # print("label's legth: ", label_length)
        cur_loss = loss_func(predict_label, label, image_length, label_length)
        print(cur_loss)
        train_avg_loss = (train_avg_loss * batch + cur_loss.item()) / (batch + 1)

        optimizer.zero_grad()
        cur_loss.backward(retain_graph=True)
        optimizer.step()

    log_file = open(os.path.join(checkpoints_dir, "log.txt"), "a+")
    log_file.write("Epoch %d | avg_loss = %.3f\n" % (epoch, train_avg_loss))
    log_file.flush()
    log_file.close()

    return train_avg_loss


def val(dataloader, model, loss_func):
    model.eval()
    val_avg_loss = 0.0
    with torch.no_grad():
        for batch, (image, image_length, label, label_length) in enumerate(dataloader):
            print("batch: ", batch)
            # 前向传播
            image = torch.unsqueeze(image, dim=1)
            image, label = image.to(device), label.to(device)
            predict_label = model(image)
            # batch, seq_len(fixed), nun_classes -> seq_len(fixed), batch, nun_classes
            predict_label = predict_label.permute(1, 0, 2)

            cur_loss = loss_func(predict_label, label, image_length, label_length)
            print(cur_loss)
            train_avg_loss = (train_avg_loss * batch + cur_loss.item()) / (batch + 1)

        log_file = open(os.path.join(checkpoints_dir, "log.txt"), "a+")
        log_file.write("Epoch %d | avg_loss = %.3f\n" % (epoch, val_avg_loss))
        log_file.flush()
        log_file.close()

    return val_avg_loss


# 开始训练
if __name__ == "__main__":

    epochs = 100
    best_loss = 100.0

    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    for epoch in range(epochs):
        print("\nepoch: %d" % (epoch + 1))
        with open(os.path.join(checkpoints_dir, "log.txt"), "a+") as log_file:
            # strftime()，格式化输出时间
            localtime = datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )  # 打印训练时间
            log_file.write(localtime)
            log_file.write(
                "\n======================training epoch %d======================\n"
                % (epoch + 1)
            )

        t1 = time.time()
        train_loss = train(train_dataloader, model, loss_func, optimizer)
        t2 = time.time()

        print("Training consumes %.2f second" % (t2 - t1))
        with open(os.path.join(checkpoints_dir, "log.txt"), "a+") as log_file:
            log_file.write("Training consumes %.2f second\n" % (t2 - t1))

        with open(os.path.join(checkpoints_dir, "log.txt"), "a+") as log_file:
            log_file.write(
                "\n======================validate epoch %d======================\n"
                % (epoch + 1)
            )
        t1 = time.time()
        val_loss = val(val_dataloader, model, loss_func)
        t2 = time.time()

        # 更新学习率
        lr_scheduler.step()
        print(epoch, lr_scheduler.get_last_lr()[0])

        print("Validation consumes %.2f second" % (t2 - t1))
        with open(os.path.join(checkpoints_dir, "log.txt"), "a+") as log_file:
            log_file.write("Validation consumes %.2f second\n\n" % (t2 - t1))

        # 保存最好的模型权重
        if val_loss < best_loss:
            best_loss = val_loss
            print("save best model")
            torch.save(model.state_dict(), "checkpoints/best_model_2022_04_23.pt")
    print("The train has done!!!")
