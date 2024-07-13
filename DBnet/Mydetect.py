# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/26 12:26
# @Author  : Jis-Baos
# @File    : Mydetect.py
import os

import cv2
import torch
import numpy as np
from PIL import Image

from MyNet import MyModel
from Myutils import shrink_out

# 训练数据存放路径
data_dir = "D:\\PythonProject\\DBnet_pytorch\\data_test"
# 权重存放路径
checkpoints_dir = "D:\\PythonProject\\DBnet_pytorch\\checkpoints"

# 如果有显卡，则转移到GPU进行训练
device = "cuda" if torch.cuda.is_available() else "cpu"

# 调用在MyNet.py中定义好的模型并加载对应的训练权重
model = MyModel()
model.load_state_dict(torch.load("D:\\PythonProject\\DBnet_pytorch\\checkpoints\\best_model.pt"))

if __name__ == '__main__':
    model.eval()
    with torch.no_grad():
        image_list = os.listdir(data_dir)
        for image in image_list:
            image_path = os.path.join(data_dir, image)
            img = Image.open(image_path).convert('RGB')

            # img = img.resize(size=(224, 224), resample=2, box=None, reducing_gap=None)
            new_width = int(img.width / 32) * 32
            new_height = int(img.height / 32) * 32
            img = img.resize(size=(new_width, new_height), resample=2)

            img = np.array(img)
            img = np.array(img / 255, dtype=np.float32)
            img_torch = torch.tensor(img, dtype=torch.float)
            img_torch = torch.permute(img_torch, (2, 0, 1))
            img_torch = torch.unsqueeze(img_torch, dim=0)
            print("img's size: ", img_torch.size())

            probability_map, threshold_map, approximate_binary_map = model(img_torch)

            # real_probability_map = torch.permute(probability_map[0], (1, 2, 0))
            # real_threshold_map = torch.permute(threshold_map[0], (1, 2, 0))
            # real_approximate_binary_map = torch.permute(approximate_binary_map[0], (1, 2, 0))
            #
            # real_probability_map = np.array(real_probability_map)
            # real_threshold_map = np.array(real_threshold_map)
            # real_approximate_binary_map = np.array(real_approximate_binary_map)
            #
            # real_probability_map = np.array(real_approximate_binary_map * 255, dtype=np.uint8)
            # real_threshold_map = np.array(real_threshold_map * 255, dtype=np.uint8)
            # real_approximate_binary_map = np.array(real_approximate_binary_map * 255, dtype=np.uint8)
            #
            # print(real_probability_map.shape)
            # print(real_threshold_map.shape)
            # print(real_approximate_binary_map.shape)
            # cv2.imshow("real_probability_map", real_probability_map)
            # cv2.imshow("real_threshold_map", real_threshold_map)
            # cv2.imshow("real_approximate_binary_map", real_approximate_binary_map)
            # cv2.waitKey(0)

            # 进行反归一化，用于显示图片
            img = np.array(img)
            img = np.array(img * 255, dtype=np.uint8)

            # 将approximate_binary_map先转化为ndarray类型，再进行二值化
            approximate_binary_map_numpy = torch.permute(approximate_binary_map[0], (1,2,0))
            approximate_binary_map_numpy = approximate_binary_map_numpy.numpy()
            print(approximate_binary_map_numpy.shape)
            approximate_binary_map_numpy[approximate_binary_map_numpy > 0.2] = 1
            approximate_binary_map_numpy[approximate_binary_map_numpy <= 0.2] = 0
            # 进行反归一化，用于显示图片
            approximate_binary_map_numpy = (approximate_binary_map_numpy * 255).astype(np.uint8)

            # 寻找approximate_binary_map中的轮廓
            contours, _ = cv2.findContours(approximate_binary_map_numpy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            pre_boxes = []
            for i in range(len(contours)):
                # 实际测试中这一步是不需要剔除多余维度的
                contour = contours[i]
                # contour = contours[i].squeeze(dim=1)
                # 用于计算封闭轮廓的周长
                contour_length = cv2.arcLength(contour, closed=True)
                # 用于计算封闭轮廓的面积
                contour_area = cv2.contourArea(contour, oriented=False)
                # 过小的可能是噪点，删除
                if contour_length > 10:
                    # cv2.minAreaRect()获取点集的最小外接矩形
                    # 返回值rect内包含该矩形的中心点坐标、高度宽度及倾斜角度等信息
                    # 使用cv2.boxPoints()可获取该矩形的四个顶点坐标
                    bounding_box = cv2.minAreaRect(contour)
                    # 按照横坐标有小到大排列
                    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
                    # 将坐标以左上角为起点，顺时针循环排列
                    if points[1][1] > points[0][1]:
                        index_1, index_4 = 0, 1
                    else:
                        index_1, index_4 = 1, 0
                    if points[3][1] > points[2][1]:
                        index_2, index_3 = 2, 3
                    else:
                        index_2, index_3 = 3, 2
                    points = [points[index_1], points[index_2], points[index_3], points[index_4]]

                    # 对approximate_binary_map的轮廓先进行放大
                    # 在寻找最小外接矩形
                    points = np.array(points)
                    box = shrink_out(points, rate=1.5 * contour_area / contour_length)
                    bounding_box2 = cv2.minAreaRect(box)
                    points = sorted(list(cv2.boxPoints(bounding_box2)), key=lambda x: x[0])
                    if points[1][1] > points[0][1]:
                        index_1, index_4 = 0, 1
                    else:
                        index_1, index_4 = 1, 0
                    if points[3][1] > points[2][1]:
                        index_2, index_3 = 2, 3
                    else:
                        index_2, index_3 = 3, 2
                    box = [points[index_1], points[index_2], points[index_3], points[index_4]]

                    # 将每一个文本实例的最小外接矩形append到pre_boxes
                    box = np.array(box).astype(np.int32)
                    pre_boxes.append(box)

            # 遍历每一个文本实例的最小外接矩形
            for i in range(len(pre_boxes)):
                box = pre_boxes[i]
                for j in range(len(box)):
                    cv2.line(img, pt1=(box[j][0], box[j][1]),
                             pt2=(box[(j + 1) % 4][0], box[(j + 1) % 4][1]), color=(0, 0, 255), thickness=2)
            cv2.imshow('predict', img)
            cv2.waitKey(0)
