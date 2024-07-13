# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/29 20:20
# @Author  : Jis-Baos
# @File    : CTCloss1.py
# @Link    : https://zhuanlan.zhihu.com/p/41674645
# @Link    : https://zhuanlan.zhihu.com/p/285918756
# @Link    : https://blog.csdn.net/jackytintin/article/details/79425866
# @Link    : https://github.com/DingKe/ml-tutorial/blob/master/ctc/CTC.ipynb

import numpy as np
from CTCutils import softmax


# 前向传播
def forward(log_probs, labels_padded):
    seq_len, embedding_size = log_probs.shape
    labels_padded_length = len(labels_padded)
    log_alpha = np.zeros([seq_len, labels_padded_length])

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
            # if: i = 1, j = 0
            # log_alpha[1][0] = log_alpha[0][0] * log_probs[1][alpha='0']
            # if: i = 1, j = 1
            # log_alpha[1][1] = (log_alpha[0][1] + log_alpha[0][0]) * log_alpha[0][alpha='labels_padded[1]']
            # if: i = 1, j = 2 and label_padded[2] = blank
            # log_alpha[1][2] = ()
            # 首元素直接用前一帧的结果
            # 之后的元素（不论字母还是blank）先求两项的和
            # 对于是字母的元素需要求三项的和，但是需要判断它和他前一个的前一个是否一样，如果一样则不加第三项
            # 相当于在相同的字母之间强制性的加入了blank
            if j - 1 >= 0:
                temp += log_alpha[i - 1, j - 1]
            if j - 2 >= 0 and alpha_encoder != 0 and alpha_encoder != labels_padded[j - 2]:
                temp += log_alpha[i - 1, j - 2]

            log_alpha[i, j] = temp * log_probs[i, alpha_encoder]

    return log_alpha


# 后向传播
def backward(log_probs, labels_padded):
    seq_len, embedding_size = log_probs.shape
    labels_padded_length = len(labels_padded)
    log_beta = np.zeros([seq_len, labels_padded_length])

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
                temp += log_beta[i + 1, j + 1]
            if j + 2 < labels_padded_length and alpha_encoder != 0 and alpha_encoder != labels_padded[j + 2]:
                temp += log_beta[i + 1, j + 2]

            log_beta[i, j] = temp * log_probs[i, alpha_encoder]

    return log_beta


# 梯度求解（输入已进行softmax）
def gradient(log_probs, labels_padded):
    seq_len, embedding_size = log_probs.shape
    labels_padded_length = len(labels_padded)

    log_alpha = forward(log_probs, labels_padded)
    log_beta = backward(log_probs, labels_padded)
    # 利用前向传播计算似然函数
    probability = log_alpha[-1, -1] + log_alpha[-1, -2]

    log_probs_grad = np.zeros([seq_len, embedding_size])
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
                log_probs_grad[m, n] += log_alpha[m, i] * log_beta[m, i]
            #  在时刻m下：计算出初步的log_probs_grad后还要除以当前元素在log_probs中概率的平方
            log_probs_grad[m, n] /= log_probs[m, n] ** 2

    # 因为优对似然函数取了对数，所以最后的结果都要除以probability
    log_probs_grad /= probability
    return log_probs_grad


# 梯度检查，验证前向、后向算法的合理性
def check_grad(log_probs, labels_padded, seq_len_index=0, embedding_size_index=0):
    # 获取log_probs_grad中最后一个时刻、最后一个类别字符的梯度
    log_probs_grad_1 = gradient(log_probs, labels_padded)[seq_len_index, embedding_size_index]

    delta = 1e-10
    # 获取log_probs中最后一个时刻、最后一个类别字符的概率
    # 此处的主要目的是将其拷贝一份，因为后续会修改这个值
    original = log_probs[seq_len_index, embedding_size_index]

    # 这一段代码的本质思想就是将每一个时刻、每一个类别字符的概率值加上无穷小，计算最后的似然函数
    # 再将加上无穷小求得的似然函数减去原本的似然函数后除以增长量（无穷小），即可获得想要的梯度，也就是变化率
    # 将原本的 log_probs 中的每个元素都加上delta
    log_probs[seq_len_index, embedding_size_index] = original + delta
    log_alpha = forward(log_probs, labels_padded)
    # 似然函数
    probability_1 = np.log(log_alpha[-1, -1] + log_alpha[-1, -2])

    # 将原本的 log_probs 中的每个元素都减去delta
    log_probs[seq_len_index, embedding_size_index] = original - delta
    log_alpha = forward(log_probs, labels_padded)
    # 似然函数
    probability_2 = np.log(log_alpha[-1, -1] + log_alpha[-1, -2])

    log_probs[seq_len_index, embedding_size_index] = original

    # 导数的定义式
    log_probs_grad_2 = (probability_1 - probability_2) / (2 * delta)
    # if np.abs(log_probs_grad_1 - log_probs_grad_2) > toleration:
    print('[%d, %d]：%.2e' % (seq_len_index, embedding_size_index, np.abs(log_probs_grad_1 - log_probs_grad_2)))


# CTC和softmax分开计算梯度
def gradient_logits_naive(net_out, labels_padded):
    log_probs_grad = gradient(net_out, labels_padded)
    # sum_log_probs_grad = np.sum(log_probs_grad * net_out, axis=1, keepdims=True)
    # net_out_grad = net_out * (log_probs_grad - sum_log_probs_grad)
    # 直接由两者的公式推算而来，相较作者的代码更加简介
    net_out_grad = log_probs_grad * net_out - net_out
    return net_out_grad


# CTC和softmax合并计算梯度
def gradient_logits(net_out, labels_padded):
    seq_len, embedding_size = net_out.shape
    labels_padded_length = len(labels_padded)

    log_alpha = forward(net_out, labels_padded)
    log_beta = backward(net_out, labels_padded)
    # 利用前向传播计算似然函数
    probability = log_alpha[-1, -1] + log_alpha[-1, -2]

    net_out_grad = np.zeros([seq_len, embedding_size])
    # 遍历时间序列
    for m in range(seq_len):
        # 遍历类别序列
        for n in range(embedding_size):
            lab = [i for i, c in enumerate(labels_padded) if c == n]
            for i in lab:
                net_out_grad[m, n] += log_alpha[m, i] * log_beta[m, i]
            net_out_grad[m, n] /= net_out[m, n] * probability
    # 因为似然函数我们只是取了对数，没有区负，所以这里为net_out_grad = net_out_grad - net_out
    net_out_grad -= net_out
    return net_out_grad


if __name__ == '__main__':

    print("**********************************init the log_probs**********************************")
    net_input = np.random.random([12, 5])  # T x m
    log_probs = softmax(net_input)
    print(log_probs)
    # 验证softmax是否成立
    print(log_probs.sum(1, keepdims=True))

    print("**********************************forward propagation**********************************")
    labels_padded = [0, 3, 0, 3, 0, 4, 0]  # 0 for blank
    log_alpha = forward(log_probs, labels_padded)
    print(log_alpha)
    # 验证，因为先前我们已经规定了传播的方式，所以最后以labels中最后两个字符结尾的一定就是我们想要的路径
    # probability = log_alpha[-1, labels[-1]] + alpha[-1, labels[-2]]
    # 似然函数
    probability = log_alpha[-1, -1] + log_alpha[-1, -2]
    print(log_alpha[-1, -1])
    print(log_alpha[-1, -2])
    print(probability)

    print("**********************************backward propagation**********************************")
    log_beta = backward(log_probs, labels_padded)
    print(log_beta)
    # 验证，因为先前我们已经规定了传播的方式，所以最后以labels中最后两个字符结尾的一定就是我们想要的路径
    # probability = log_beta[0, labels[0]] + alpha[0, labels[1]]
    # 似然函数
    probability = log_beta[0, 0] + log_beta[0, 1]
    print(log_beta[0, 0])
    print(log_beta[0, 1])
    print(probability)

    print("**********************************calculate log_probs_grad**********************************")
    log_probs_grad = gradient(log_probs, labels_padded)
    print(log_probs_grad)

    print("*************************将基于前向-后向算法得到梯度与基于数值的梯度比较*************************")
    for seq_len_index in range(log_probs.shape[0]):
        for embedding_size_index in range(log_probs.shape[1]):
            check_grad(log_probs, labels_padded, seq_len_index, embedding_size_index)

    print("***************************CTCloss和softmax一起求导和分步求导的验证***************************")
    net_out_grad1 = gradient_logits_naive(log_probs, labels_padded)
    net_out_grad2 = gradient_logits(log_probs, labels_padded)
    print(net_out_grad1)
    print(net_out_grad2)
    # 比较两种计算方法的差异，误差极小，证明了算法的有效性
    print(np.sum(np.abs(net_out_grad1 - net_out_grad2)))