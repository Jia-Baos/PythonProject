# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/29 20:20
# @Author  : Jis-Baos
# @File    : CTCutils.py
# @Link    : https://zhuanlan.zhihu.com/p/41674645
# @Link    : https://zhuanlan.zhihu.com/p/285918756
# @Link    : https://blog.csdn.net/jackytintin/article/details/79425866
# @Link    : https://github.com/DingKe/ml-tutorial/blob/master/ctc/CTC.ipynb

import numpy as np
negative_infinity = -float('inf')


def softmax(logits):
    max_value = np.max(logits, axis=1, keepdims=True)
    # 此步操作使得数值全部为负，进而计算指数时使得结果处于区间[0, 1]内
    exp = np.exp(logits - max_value)
    exp_sum = np.sum(exp, axis=1, keepdims=True)
    dist = exp / exp_sum
    return dist


# log域计算函数
def log_sum_exp(a, b):
    """
      log(a + b)
    = log(a * (1 + b/a))
    = log(a) + log(1 + b/a)
    = log(a) + log(1 + exp(log(b/a)))
    = log(a) + log(1 + exp(log(b) - log(a)))
    """
    # 使得a始终为较大者
    if a < b:
        a, b = b, a
    if b == negative_infinity:
        return a
    else:
        return a + np.log(1 + np.exp(b - a))


# 编码时插入空白
def insert_blank(labels, blank=0):
    new_labels = [blank]
    for l in labels:
        new_labels += [l, blank]
    return new_labels


# 去除blank，并删除空格间的重复元素
def remove_blank(labels, blank=0):
    temp_labels = []

    # 将两个blank之间相邻的重复元素剔除的只剩下一个
    previous = None
    for item in labels:
        if item != previous:
            temp_labels.append(item)
            previous = item

    # 删除空白
    new_labels = [item for item in temp_labels if item != blank]

    return new_labels