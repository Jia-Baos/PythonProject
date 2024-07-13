import math
import os

import numpy as np
import torch
import torch.nn as nn
from Myutils import shrink_out

# With square kernels and equal stride
# stride=2，interpolate -> padding -> conv
# 5 * 5 -> 11 * 11 -> 13 * 13 -> 11 * 11
m1 = nn.ConvTranspose2d(2, 4, kernel_size=3, stride=2, padding=1)
# non-square kernels and unequal stride and with padding
# https://blog.csdn.net/qq_41368247/article/details/86626446?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3.pc_relevant_paycolumn_v3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3.pc_relevant_paycolumn_v3&utm_relevant_index=6
m2 = nn.ConvTranspose2d(2, 4, kernel_size=(3, 5), stride=(2, 1), padding=(1, 2))
input = torch.randn((1, 2, 5, 5), dtype=torch.float)
output1 = m1(input)
output2 = m2(input)
print(output1.size())
print(output2.size())

# exact output size can be also specified as an argument
# input = torch.randn(1, 16, 12, 12)
# downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
# upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
# h = downsample(input)
# h.size()
# torch.Size([1, 16, 6, 6])
# output = upsample(h, output_size=input.size())
# output.size()
# torch.Size([1, 16, 12, 12])

import cv2

img = cv2.imread("D:\\PythonProject\\DBnet_pytorch\\data_test\\test.png")
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1[img1 > 180] = 255
img1[img1 < 120] = 0
contours, _ = cv2.findContours(img1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print("contours: ", contours)
pre_boxes = []
for i in range(len(contours)):
    print("item in contours: ", contours[i])
    contour = contours[i]
    # 用于计算封闭轮廓的周长或曲线的长度
    contour_length = cv2.arcLength(contour, closed=True)
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
        box = shrink_out(points, rate=2.0)
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
