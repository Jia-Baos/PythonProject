# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/29 20:20
# @Author  : Jis-Baos
# @File    : CTCloss2.py
# @Link    : https://zhuanlan.zhihu.com/p/41674645
# @Link    : https://zhuanlan.zhihu.com/p/285918756
# @Link    : https://blog.csdn.net/jackytintin/article/details/79425866
# @Link    : https://github.com/DingKe/ml-tutorial/blob/master/ctc/CTC.ipynb

import numpy as np
from CTCutils import softmax, log_sum_exp, negative_infinity


# 前向传播（log优化版）
def forward_log(log_probs, labels_padded):
    seq_len, embedding_size = log_probs.shape
    labels_padded_length = len(labels_padded)
    log_alpha = np.ones([seq_len, labels_padded_length]) * negative_infinity

    # 初始化0时刻的路径，即初始路径，后面的路径按照设定的规则进行规划
    log_alpha[0, 0] = log_probs[0, labels_padded[0]]
    log_alpha[0, 1] = log_probs[0, labels_padded[1]]

    # 遍历时间序列
    for i in range(1, seq_len):
        # 遍历已经用blank填充为 2 * labels_length + 1 的labels序列，即labels_padded
        for j in range(labels_padded_length):
            # 通过alpha_encoder去访问时刻i，字符alpha发生的概率，注意此时的alpha_encoder已经是对字母编码后的结果
            alpha_encoder = labels_padded[j]
            temp = log_alpha[i - 1, j]
            # 首元素直接用前一帧的结果
            # 之后的元素（不论字母还是blank）先求两项的和
            # 对于是字母的元素需要求三项的和，但是需要判断它和他前一个的前一个是否一样，如果一样则不加第三项
            # 相当于在相同的字母之间强制性的加入了blank
            if j - 1 >= 0:
                temp = log_sum_exp(temp, log_alpha[i - 1, j - 1])
            if j - 2 >= 0 and alpha_encoder != 0 and alpha_encoder != labels_padded[j - 2]:
                temp = log_sum_exp(temp, log_alpha[i - 1, j - 2])

            log_alpha[i, j] = temp + log_probs[i, alpha_encoder]

    return log_alpha


# 后向传播（log优化版）
def backward_log(log_probs, labels_padded):
    seq_len, embedding_size = log_probs.shape
    labels_padded_length = len(labels_padded)
    log_beta = np.ones([seq_len, labels_padded_length]) * negative_infinity

    # 初始化 seq_len - 1 时刻的路径，即初始路径，后面的路径按照设定的规则进行规划
    log_beta[-1, -1] = log_probs[-1, labels_padded[-1]]
    log_beta[-1, -2] = log_probs[-1, labels_padded[-2]]

    # 遍历时间序列，注意需要反向遍历
    # seq_len = 11, index -> (0, 10)
    # 因为我们已经初始化了index=10时刻的路径，所以需遍历的index为(0, 9)，且为从后向前遍历
    for i in range(seq_len - 2, -1, -1):
        # print("backward i: ", i)
        # 遍历labels_padded，注意需要反向遍历
        for j in range(labels_padded_length - 1, -1, -1):
            # print("backward j: ", j)
            alpha_encoder = labels_padded[j]
            temp = log_beta[i + 1, j]
            if j + 1 < labels_padded_length:
                temp = log_sum_exp(temp, log_beta[i + 1, j + 1])
            if j + 2 < labels_padded_length and alpha_encoder != 0 and alpha_encoder != labels_padded[j + 2]:
                temp = log_sum_exp(temp, log_beta[i + 1, j + 2])

            log_beta[i, j] = temp + log_probs[i, alpha_encoder]

    return log_beta


# 梯度求解（输入已进行softmax，log优化版）
def gradient_log(log_probs, labels_padded):
    seq_len, embedding_size = log_probs.shape
    labels_padded_length = len(labels_padded)

    log_alpha = forward_log(log_probs, labels_padded)
    log_beta = backward_log(log_probs, labels_padded)
    # 利用前向传播计算似然函数
    probability = log_sum_exp(log_alpha[-1, -1], log_alpha[-1, -2])

    log_probs_grad = np.ones([seq_len, embedding_size]) * negative_infinity
    # 遍历时间序列
    for m in range(seq_len):
        # print("time: ", m)
        # 遍历类别序列，blank打头
        for n in range(embedding_size):
            # 值得注意的是，embedding_size中有些字符是不会在labels_padded中出现的
            # 那么它们也就没有必要去求梯度，直接置为0，此外还有一些0的产生是因为log_alpha与log_beta相乘结果为0
            # lab中存储的是labels_padded中相同元素（n）的index
            lab = [i for i, c in enumerate(labels_padded) if c == n]
            # print("lab: ", lab)
            for i in lab:
                # 在时刻m下：首先将log_alpha和log_beta中相同元素的概率先相乘再相加
                log_probs_grad[m, n] = log_sum_exp(log_probs_grad[m, n], log_alpha[m, i] + log_beta[m, i])
            #  在时刻m下：计算出初步的log_probs_grad后还要除以当前元素在log_probs中概率的平方
            log_probs_grad[m, n] -= 2 * log_probs[m, n]

    # 因为优对似然函数取了对数，所以最后的结果都要除以probability
    log_probs_grad -= probability
    return log_probs_grad


if __name__ == '__main__':
    print("**********************************init the log_probs**********************************")
    net_input = np.random.random([12, 5])  # T x m
    log_probs = softmax(net_input)
    print(log_probs)
    # 验证softmax是否成立
    print(log_probs.sum(1, keepdims=True))

    print("**********************************forward propagation**********************************")
    labels_padded = [0, 3, 0, 3, 0, 4, 0]  # 0 for blank
    log_alpha = forward_log(np.log(log_probs), labels_padded)
    print(np.exp(log_alpha))

    print("**********************************backward propagation**********************************")
    log_beta = backward_log(np.log(log_probs), labels_padded)
    print(np.exp(log_beta))

    print("********************************calculate log_probs_grad********************************")
    log_probs_grad = gradient_log(np.log(log_probs), labels_padded)
    print(np.exp(log_probs_grad))
