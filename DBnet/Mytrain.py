# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/24 18:19
# @Author  : Jis-Baos
# @File    : Mytrain.py

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import os
import time
import datetime

from MyNet import MyModel
from MyDataSet import MyDataSet

# 训练数据存放路径
data_dir = "D:\\PythonProject\\DBnet_pytorch\\data"
# 权重存放路径
checkpoints_dir = "D:\\PythonProject\\DBnet_pytorch\\checkpoints"

# 加载训练数据集
train_dataset = MyDataSet(data_dir, mode='train')
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

# 加载测试数据集
val_dataset = MyDataSet(data_dir, mode='test')
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

# 如果有显卡，则转移到GPU进行训练
device = "cuda" if torch.cuda.is_available() else "cpu"

# 调用在MyNet.py中定义好的模型并加载对应的训练权重
model = MyModel()
# model.load_state_dict(torch.load("D:\\PythonProject\\DBnet_pytorch\\checkpoints\\best_model.pt"))

# 损失函数
bce_loss = nn.BCELoss()
l1_loss = nn.L1Loss()

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# 定义学习率变化方案
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


# 定义训练函数
def train(dataloader, model, optimizer):
    model.train()
    train_batch_loss = 0.0
    alpha, beta = 10.0, 10.0
    for batch, data in enumerate(dataloader):
        print("batch: ", batch)
        img = data[0].to(device)
        shrink_label = data[1].to(device)
        threshold_label = data[2].to(device)
        probability_map, threshold_map, approximate_binary_map = model(img)
        loss_probability_map = bce_loss(probability_map, shrink_label)
        loss_threshold_map = l1_loss(threshold_map, threshold_label)
        loss_approximate_binary_map = bce_loss(approximate_binary_map, shrink_label)
        loss = loss_probability_map + alpha * loss_approximate_binary_map + beta * loss_threshold_map
        train_batch_loss = (train_batch_loss * batch + loss.item()) / (batch + 1)
        print("train_batch_loss: ", train_batch_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log_file = open(os.path.join(checkpoints_dir, "log.txt"), "a+")
    log_file.write("Epoch %d | avg_loss = %.3f\n" % (epoch, train_batch_loss))
    log_file.flush()
    log_file.close()

    return train_batch_loss


# 定义测试函数
def val(dataloader, model):
    model.eval()
    val_batch_loss = 0.0
    alpha, beta = 10.0, 10.0
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            print("batch: ", batch)
            img = data[0].to(device)
            shrink_label = data[1].to(device)
            threshold_label = data[2].to(device)
            probability_map, threshold_map, approximate_binary_map = model(img)
            loss_probability_map = bce_loss(probability_map, shrink_label)
            loss_threshold_map = l1_loss(threshold_map, threshold_label)
            loss_approximate_binary_map = bce_loss(approximate_binary_map, shrink_label)
            loss = loss_probability_map + alpha * loss_approximate_binary_map + beta * loss_threshold_map
            val_batch_loss = (val_batch_loss * batch + loss.item()) / (batch + 1)
            print("val_batch_loss: ", val_batch_loss)

        log_file = open(os.path.join(checkpoints_dir, "log.txt"), "a+")
        log_file.write("Epoch %d | avg_loss = %.3f\n" % (epoch, val_batch_loss))
        log_file.flush()
        log_file.close()

    return val_batch_loss


# 开始训练
if __name__ == '__main__':
    epochs = 100
    best_loss = 20

    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    for epoch in range(epochs):
        print("\n epoch: %d" % (epoch + 1))
        with open(os.path.join(checkpoints_dir, "log.txt"), "a+") as log_file:
            localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 打印训练时间
            log_file.write(localtime)
            log_file.write("\n======================training epoch %d======================\n" % (epoch + 1))

        t1 = time.time()
        train_loss = train(train_dataloader, model, optimizer)
        t2 = time.time()

        print("Training consumes %.2f second" % (t2 - t1))
        with open(os.path.join(checkpoints_dir, "log.txt"), "a+") as log_file:
            log_file.write("Training consumes %.2f second\n" % (t2 - t1))

        with open(os.path.join(checkpoints_dir, "log.txt"), "a+") as log_file:
            log_file.write("\n======================validate epoch %d======================\n" % (epoch + 1))

        t1 = time.time()
        val_loss = val(val_dataloader, model)
        t2 = time.time()

        print("Validation consumes %.2f second" % (t2 - t1))
        with open(os.path.join(checkpoints_dir, "log.txt"), "a+") as log_file:
            log_file.write("Validation consumes %.2f second\n\n" % (t2 - t1))

        # 更新学习率
        lr_scheduler.step()

        # 保存最好的模型权重
        if val_loss < best_loss:
            best_loss = val_loss
            print("save best model")
            torch.save(model.state_dict(), 'checkpoints/best_model.pt')
    print("The train has done!!!")
