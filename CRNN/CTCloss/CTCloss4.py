# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/30 19:46
# @Author  : Jis-Baos
# @File    : CTCloss4.py
# @Link    : https://zhuanlan.zhihu.com/p/41674645
# @Link    : https://zhuanlan.zhihu.com/p/285918756
# @Link    : https://blog.csdn.net/jackytintin/article/details/79425866
# @Link    : https://github.com/DingKe/ml-tutorial/blob/master/ctc/CTC.ipynb

import numpy as np
from CTCutils import softmax, remove_blank


# 束搜索（beam search）
# 该方法的核心在于每一时刻都从路径中挑出前beam_size个路径
# 虽然看起来路径数目少了，但实际进行的计算量还是非常大的
def beam_decode(log_probs, beam_size=2):
    seq_len, embedding_size = log_probs.shape
    log_log_prob = np.log(log_probs)

    # beam是由元组构成的列表，元组里面存放路径和得分
    beam_list = [([], 0)]
    for i in range(seq_len):  # for every timestep
        new_beam = []
        for prefix, score in beam_list:
            for j in range(embedding_size):  # for every state
                new_prefix = prefix + [j]
                new_score = score + log_log_prob[i, j]
                new_beam.append((new_prefix, new_score))

        # top beam_size
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam_list = new_beam[:beam_size]

    return beam_list


if __name__ == '__main__':

    net_out = np.random.random([20, 6])
    log_probs = softmax(net_out)
    beam_list = beam_decode(log_probs, beam_size=3)

    for string, score in beam_list:
        print(remove_blank(string), score)
