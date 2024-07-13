# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/24 18:19
# @Author  : Jis-Baos
# @File    : Mytrain.py

import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from MyNet import MyModel
from MyDataSet import MyDataSet

data_dir = "D:\\PythonProject\\DBnet_pytorch\\data"
checkpoints_dir = "D:\\PythonProject\\DBnet_pytorch\\checkpoints"

dataset = MyDataSet(data_dir, mode='test')
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MyModel()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


def train(dataloader, model, optimizer):
    model.train()
    loss = 0.0
    alpha, beta = 10.0, 10.0
    for step, data in enumerate(dataloader):
        img = data[0].to(device)
        shrink_label = data[1].to(device)
        threshold_label = data[2].to(device)
        probability_map, threshold_map, approximate_binary_map = model(img)
        loss_probability_map = nn.BCELoss()(probability_map, shrink_label)
        loss_threshold_map = nn.L1Loss()(threshold_map, threshold_label)
        loss_approximate_binary_map = nn.BCELoss()(approximate_binary_map, shrink_label)

        loss = loss_probability_map + alpha * loss_approximate_binary_map + beta * loss_threshold_map
        print("batch's loss: ", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss


if __name__ == '__main__':
    epochs = 20
    best_loss = 10

    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    for epoch in range(epochs):
        train_loss = train(dataloader, model, optimizer)
        print("epoch: ", epoch)
        print("loss: ", train_loss)

        # 更新学习率
        lr_scheduler.step()

        # 保存最好的模型权重
        if train_loss < best_loss:
            best_loss = train_loss
            print("save best model")
            torch.save(model.state_dict(), 'checkpoints/best_model.pt')
    print("The train has done!!!")
