# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/24 11:18
# @Author  : Jis-Baos
# @File    : MyDataSet.py
# @Link1   : https://zhuanlan.zhihu.com/p/368035566
# @Link2   : https://zhuanlan.zhihu.com/p/382641896
# @Link3   : https://link.zhihu.com/?target=https%3A//github.com/yts2020/DBnet_pytorch


import os
import cv2
import numpy as np
import torch
from PIL import Image
from Myutils import shrink, make_threshold_map
from torch.utils.data import Dataset, DataLoader

# ground truth的坐标原点在左上角
# 四个坐标点顺序为 左上 -> 右上 -> 右下 -> 左下


class MyDataSet(Dataset):
    def __init__(self, dataset_dir, mode="train", trans=None):
        self.data_dir = dataset_dir
        self.mode = mode
        self.train_data_dir = os.path.join(dataset_dir, "train_data")
        self.test_data_dir = os.path.join(dataset_dir, "test_data")

        # img_list存的只是图片的名字
        self.image_list = []
        self.label_list = []
        if mode == 'train':
            self.train_image_dir = os.path.join(self.train_data_dir, "image")
            self.train_label_dir = os.path.join(self.train_data_dir, "label")
            for image in os.listdir(self.train_image_dir):
                self.image_list.append(image)
            for label in os.listdir(self.train_label_dir):
                self.label_list.append(label)

        elif mode == 'test':
            self.test_image_dir = os.path.join(self.test_data_dir, "image")
            self.test_label_dir = os.path.join(self.test_data_dir, "label")
            for image in os.listdir(self.test_image_dir):
                self.image_list.append(image)
            for label in os.listdir(self.test_label_dir):
                self.label_list.append(label)

        self.trans = trans

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        image_path, label_path = '', ''

        if self.mode == 'train':
            image_path = os.path.join(self.train_image_dir, self.image_list[item])
            label_path = os.path.join(self.train_label_dir, self.label_list[item])

        elif self.mode == 'test':
            image_path = os.path.join(self.test_image_dir, self.image_list[item])
            label_path = os.path.join(self.test_label_dir, self.label_list[item])

        image = Image.open(image_path).convert('RGB')
        # 宽高调整为32的倍数
        new_width = int(image.width / 32) * 32
        scale_x = new_width / image.width
        new_height = int(image.height / 32) * 32
        scale_y = new_height / image.height
        image = image.resize(size=(new_width, new_height), resample=2)

        # 原txt文件为gbk编码，故需做调整
        with open(label_path, encoding='utf-8-sig', mode='r') as f:
            # 列表，txt文件中的一行构成其中的一个元素
            label = f.readlines()
            # label = int(label[0][0])

        ground_truth_boxes_all = []
        # 把列表中的ground truth框的坐标提取出来
        for i in range(len(label)):
            # 以空格为间隔将坐标分割出来
            ground_truth_data = label[i].strip().split(',')
            x_coordination_list = list([])
            x_coordination_list.append(int(int(ground_truth_data[0]) * scale_x))
            x_coordination_list.append(int(int(ground_truth_data[2]) * scale_x))
            x_coordination_list.append(int(int(ground_truth_data[4]) * scale_x))
            x_coordination_list.append(int(int(ground_truth_data[6]) * scale_x))
            y_coordination_list = list([])
            y_coordination_list.append(int(int(ground_truth_data[1]) * scale_y))
            y_coordination_list.append(int(int(ground_truth_data[3]) * scale_y))
            y_coordination_list.append(int(int(ground_truth_data[5]) * scale_y))
            y_coordination_list.append(int(int(ground_truth_data[7]) * scale_y))

            ground_truth_boxes = list([])
            # 将多边形每个点的横、纵坐标封装成一个小数组，再将这些小数组封装成一个大数组
            for i in range(len(x_coordination_list)):
                ground_truth_boxes.append([x_coordination_list[i], y_coordination_list[i]])
            # 将每一个ground truth的大数组继续封装起来，形成一个大大数组
            ground_truth_boxes_all.append(ground_truth_boxes)
        ground_truth_boxes_all = np.array(ground_truth_boxes_all, dtype=np.int32)

        # probability_map是一通道的
        probability_map = np.zeros(shape=(image.height, image.width), dtype=np.float32)
        # 遍历每一个候选框
        for gt_boxes in ground_truth_boxes_all:
            poly = shrink(gt_boxes, 0.4)
            # 填充任意形状的图形，这里还要再次恢复ploy的维度（3）
            # 也许这里可以直接
            cv2.fillPoly(probability_map, [poly], (255.0, 255.0, 255.0))

        # 将image转化为ndarray
        img = np.array(image)
        # 阈值图在生成的时候已经进行了归一化？Yes，原函数中归一化到归一化到0.3到0.7之内
        threshold_map = make_threshold_map(img=img, text_polys=ground_truth_boxes_all)
        # 此处正常显示阈值图
        cv2.imshow("temp", threshold_map)

        # img dimension: 224 * 224 * 3 -> 3 * 224 * 224
        img = np.transpose(img, (2, 0, 1))
        img = np.array(img / 255, dtype=np.float32)

        probability_map = np.array(probability_map / 255, dtype=np.float32)
        # threshold_map = np.array(threshold_map / 255, dtype=np.float32)

        print(img.dtype)
        print(probability_map.dtype)
        print(threshold_map.dtype)

        probability_map = np.expand_dims(probability_map, axis=0)
        threshold_map = np.expand_dims(threshold_map, axis=0)

        img = torch.tensor(img, dtype=torch.float)
        probability_map = torch.tensor(probability_map, dtype=torch.float)
        threshold_map = torch.tensor(threshold_map, dtype=torch.float)

        return img, probability_map, threshold_map


if __name__ == '__main__':
    data_dir = "D:\\PythonProject\\DBnet_pytorch\\mydata"
    dataset = MyDataSet(data_dir, mode='train')
    # DataLoader要求输入图片大小一致
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    for real_data in dataloader:
        real_img, real_probability_map, real_threshold_map = real_data
        print(type(real_img))
        print(type(real_probability_map))
        print(type(real_threshold_map))

        real_img = real_img[0]
        real_probability_map = real_probability_map[0]
        real_threshold_map = real_threshold_map[0]
        print(real_img.size())
        print(real_probability_map.size())
        print(real_threshold_map.size())

        real_img = real_img.numpy()
        real_probability_map = real_probability_map.numpy()
        real_threshold_map = real_threshold_map.numpy()

        print(real_img.dtype)
        print(real_probability_map.dtype)
        print(real_threshold_map.dtype)

        # real_img = np.array(real_img * 255, dtype=np.uint8)
        # real_probability_map = np.array(real_probability_map * 255, dtype=np.uint8)
        # real_threshold_map = np.array(real_threshold_map * 255, dtype=np.uint8)

        show_real_img = cv2.merge([real_img[0], real_img[1], real_img[2]])
        show_real_probability_map = cv2.merge([real_probability_map[0], real_probability_map[0], real_probability_map[0]])
        show_real_threshold_map = cv2.merge([real_threshold_map[0], real_threshold_map[0], real_threshold_map[0]])
        cv2.imshow("test1", show_real_img)
        cv2.imshow("test2", show_real_probability_map)
        cv2.imshow("test3", show_real_threshold_map)
        cv2.waitKey(0)

