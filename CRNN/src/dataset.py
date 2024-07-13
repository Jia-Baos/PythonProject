# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/27 20:51
# @Author  : Jis-Baos
# @File    : MyDataSet.py

import os
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from dict_tools import CHAR_LIST, MAX_LENGTH, encode, decode


def add_elements(s, element, count):
    return s + "".join(element for _ in range(count))


class MyDataSet(Dataset):
    def __init__(self, dataset_dir, mode="train", trans=None):
        self.data_dir = dataset_dir
        self.mode = mode
        self.trans = trans

        self.image_dir = os.path.join(dataset_dir, "data")
        self.label_dir = os.path.join(dataset_dir, "label")

        # img_list存的只是图片的名字
        self.img_list = []
        for filename in os.listdir(self.image_dir):
            self.img_list.append(filename.split(".")[0])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = os.path.join(self.image_dir, self.img_list[item] + ".png")
        image = np.array(Image.open(img_path).convert("L"))

        h, w = image.shape

        image = cv2.resize(image, (100, 32))

        # 标准化
        image = image.astype(np.float32) / 255.0
        image -= 0.5
        image /= 0.5  # H*W

        label_path = os.path.join(self.label_dir, self.img_list[item] + ".txt")

        with open(label_path, "r") as file:
            label = file.readline().replace("\n", "")
            label_length = len(label)
            indices = encode(label)

        image_length = w
        image = torch.tensor(image, dtype=torch.float)

        target_length = label_length
        target = np.zeros(MAX_LENGTH)
        target[:label_length] = indices
        target = torch.tensor(target, dtype=torch.float)

        return image, image_length, target, target_length


if __name__ == "__main__":
    data_dir = "E:/OCRData/ch4_training_word_images_gt"
    dataset = MyDataSet(data_dir, mode="train")
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False
    )

    max_length = 0
    for batch, data_couple in enumerate(dataloader):
        print("batch: ", batch)
        image, image_length, label, label_length = data_couple
        print(image)
        print(label)
        print(image.shape)
        print(label.shape)
        print(image_length.shape)
        print(label_length.shape)

        length = label_length.squeeze().numpy().astype(int)
        print("---------", length, max_length, type(length), type(max_length))
        if length > max_length:
            max_length = length
        print("max_length in dataset", max_length)

        # image_np = image.numpy()
        # print("-----------")
        # print("image's size: ", image.shape)
        # print("image's dtyep: ", image.dtype)
        # print("image's length: ", image_length)

        # print("label: ", label)
        # print("label's length: ", label_length)

        # # 测试解码时保证 batchsize 为1
        # text = decode(
        #     label.squeeze().numpy().astype(int),
        #     label_length.squeeze().numpy().astype(int),
        # )
        # print(text)
