# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/29 20:20
# @Author  : Jis-Baos
# @File    : CTCloss3.py
# @Link    : https://zhuanlan.zhihu.com/p/41674645
# @Link    : https://zhuanlan.zhihu.com/p/285918756
# @Link    : https://blog.csdn.net/jackytintin/article/details/79425866
# @Link    : https://github.com/DingKe/ml-tutorial/blob/master/ctc/CTC.ipynb

import numpy as np
from CTCutils import softmax, remove_blank


# 贪心算法搜索
def greedy_decode(log_pros, blank=0):
    # 每一时刻都取最大概率下的理想最优路径
    optimal_path = np.argmax(log_pros, axis=1)
    predicted_string = remove_blank(optimal_path, blank)
    return optimal_path, predicted_string


if __name__ == '__main__':
    net_out = np.random.random([20, 6])
    log_probs = softmax(net_out)
    # print(log_probs)

    optimal_path, predicted_string = greedy_decode(log_probs)
    print("optimal_path's size: ", len(optimal_path))
    print(optimal_path)
    print("predicted_string's size: ", len(predicted_string))
    print(predicted_string)