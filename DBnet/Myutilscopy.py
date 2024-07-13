# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/26 15:48
# @Author  : Jis-Baos
# @File    : Myutilscopy.py
import cv2
import copy
import pyclipper
import numpy as np
import Polygon as plg
from shapely.geometry import Polygon


def dist(a, b):
    """
    计算距离
        :param a: ground truth box中的一个顶点
        :param b: ground truth box中与顶点a连接的下一个顶点b
        :return: 两点之间的直线距离
    """
    return np.sqrt(np.sum((a - b) ** 2))


def perimeter(bbox):
    """
    计算周长
        :param bbox: 图片中一个实例的ground truth box
        :return: 多边形的周长
    """
    peri = 0.0
    # 遍历多边形所有的顶点
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bbox, rate):
    """
    将标签轮廓向内缩小，这里调用了Pyclipper库中的Vatti clipping algorithm算法来实现
    多边形的缩小
        :param bbox: 图片中一个实例的ground truth box
        :param rate: 缩放因子
        :return: 缩小后的ground truth box
    """
    area = plg.Polygon(bbox).area()     # 面积
    peri = perimeter(bbox)  # 周长
    # 初始化PyclipperOffset类，这里的详细内容参考函数中提供的链接网址
    pco = pyclipper.PyclipperOffset()
    # 将bbox初始化为路径，从后面的函数来看，路径似乎被嵌入了pco中
    pco.AddPath(bbox, join_type=pyclipper.JT_ROUND, end_type=pyclipper.ET_CLOSEDPOLYGON)
    d = int(area * (1 - rate * rate) / peri)
    # 通过对text polygon进行-D的缩小获取到正区域G_d
    # 直接对嵌入pco中的路径进行放缩，之后返回放缩的路径，注意其维度为3
    shrink_bbox = pco.Execute(-d)
    shrink_bbox = np.array(shrink_bbox)[0]
    return shrink_bbox


# 将预测结果往外扩
def shrink_out(bbox, rate):
    area = plg.Polygon(bbox).area()
    peri = perimeter(bbox)
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    d = int(area * rate / peri)
    shrink_bbox = pco.Execute(d)
    shrink_bbox = np.array(shrink_bbox)[0]
    return shrink_bbox


def distance_matrix(xs, ys, a, b):
    """
    以矩阵方式计算点到线段的距离
        :param xs: dimension -> padded_polygon最大外接矩形的height、width
        :param ys: dimension -> padded_polygon最大外接矩形的height、width
        :param a: ground truth box中的顶点a
        :param b: ground truth box中与顶点a连接的下一个顶点b
        :return:
    """
    # 获取点a、点b的横、纵坐标
    x1, y1 = a[0], a[1]
    x2, y2 = b[0], b[1]
    # 向量a -> s、向量a -> b，两个向量作点乘，即as * ab = |as||ab|cos(theta)
    u1 = (((xs - x1) * (x2 - x1)) + ((ys - y1) * (y2 - y1)))
    # 将点乘结果除以|ab|的平方得到：|as|cos(theta)/|ab|
    u = u1 / (np.square(x1 - x2) + np.square(y1 - y2))
    # 为什么元素小于0就要设置为2？其只是为了后面能够将这块区域给检测出来，用正确的距离来做替换
    u[u <= 0] = 2
    # 由向量a -> s的顶点s向向量a -> b作垂线，ix为垂足的横坐标
    ix = x1 + u * (x2 - x1)
    # 由向量a -> s的顶点s向向量a -> b作垂线，iy为垂足的纵坐标
    iy = y1 + u * (y2 - y1)
    # 这里求得的是向量a -> s的顶点s向向量a -> b所作垂线的距离
    distance = np.sqrt(np.square(xs - ix) + np.square(ys - iy))
    distance1 = distance.copy()
    # 直接计算矩形区域内所有点到a点和b点的距离，之后取其中的较小者，用来替换a点左侧和b点右侧区间里错误的距离
    distance2 = np.sqrt(np.fmin(np.square(xs - x1) + np.square(ys - y1), np.square(xs - x2) + np.square(ys - y2)))
    # 为什么要有这一步，因为之前我们的计算公式在a点左侧和b点右侧失效了，所以用上一步的计算结果将其修正
    distance1[u >= 1] = distance2[u >= 1]
    return distance1


def draw_border_map(polygon, canvas, mask, shrink_ratio):
    """
    计算点到各线段的最小距离
        :param polygon: 图片中一个实例的ground truth box
        :param canvas: threshold_map
        :param mask: mask
        :param shrink_ratio: 缩放因子
        :return:
    """
    # 对图片中实例的ground truth box的维度和横纵坐标是否存在进行确认
    polygon = np.array(polygon)
    assert polygon.ndim == 2
    assert polygon.shape[1] == 2

    # 用Polygon函数来计算面积
    polygon_shape = Polygon(polygon)
    if polygon_shape.area <= 0:
        return

    # 计算: D = A * (1 - r ^ 2) / L
    # 通过对text polygon进行+D的放大获取到G_s -> threshold map的label
    # 我们把G_s和G_d之间的间隔当作文本区域的边界
    distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length

    # 转换类型，将每一个点的横、纵坐标转化为元组，此步转化可以忽略
    # subject = [tuple(l) for l in polygon]
    polygon_deepcopy = copy.deepcopy(polygon)
    polygon_dilate = pyclipper.PyclipperOffset()
    polygon_dilate.AddPath(polygon_deepcopy, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

    # 提取多余的维度后转化为ndarray，但是这里并不是返回四个点，而是多个，why?
    # dilate后原来的一个点扩充为三个点
    padded_polygon = np.array(polygon_dilate.Execute(distance)[0])

    # 在mask上填充dilate后的区域
    cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], (255.0))

    # 用于获取dilate后区域的最大外接矩形的shape
    x_min = padded_polygon[:, 0].min()
    x_max = padded_polygon[:, 0].max()
    y_min = padded_polygon[:, 1].min()
    y_max = padded_polygon[:, 1].max()
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    # 将polygon平移到dilate后（padded_polygon）的最大外接矩形框内，是要在这个矩形框里面计算什么吗？
    polygon[:, 0] = polygon[:, 0] - x_min
    polygon[:, 1] = polygon[:, 1] - y_min

    # np.linspace()用于生成一个等差分布的序列，之后在将其复制height份，合并成二维数组
    xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
    ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

    # dimension = 3
    distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)

    # 找到最小的距离
    for i in range(polygon.shape[0]):
        j = (i + 1) % polygon.shape[0]
        absolute_distance = distance_matrix(xs, ys, polygon[i], polygon[j])
        distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
    distance_map = np.min(distance_map, axis=0)

    # 防止坐标框越界
    x_min_valid = min(max(0, x_min), canvas.shape[1] - 1)
    x_max_valid = min(max(0, x_max), canvas.shape[1] - 1)
    y_min_valid = min(max(0, y_min), canvas.shape[0] - 1)
    y_max_valid = min(max(0, y_max), canvas.shape[0] - 1)

    # 距离原始polygon越近值越接近1，超出distance的值都为0
    # fmax，返回两个序列对应位置上的最大值，返回一个有最大值组成的新序列
    canvas[y_min_valid:y_max_valid + 1, x_min_valid:x_max_valid + 1] = np.fmax(
        1 - distance_map[
            y_min_valid - y_min:y_max_valid - y_max + height,
            x_min_valid - x_min:x_max_valid - x_max + width],
        canvas[y_min_valid:y_max_valid + 1, x_min_valid:x_max_valid + 1])


def make_threshold_map(img, text_polys, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7):
    """
    生成阈值图label
        :param img: 转化为ndarray的原图像
        :param text_polys: 图片里所有实例的ground box truth
        :param shrink_ratio: 缩放因子
        :param thresh_min: 最小阈值
        :param thresh_max: 最大阈值
        :return: threshold_map -> G_s - G_d
    """
    # threshold_map和mask都是一通道的
    threshold_map = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.float32)
    mask = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.float32)
    for i in range(len(text_polys)):
        draw_border_map(text_polys[i], threshold_map, mask=mask, shrink_ratio=shrink_ratio)

    threshold_map = threshold_map * (thresh_max - thresh_min) + thresh_min  # 归一化到0.3到0.7之内
    return threshold_map
