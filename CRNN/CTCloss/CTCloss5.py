# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/30 20:23
# @Author  : Jis-Baos
# @File    : CTCloss5.py
# @Link    : https://zhuanlan.zhihu.com/p/41674645
# @Link    : https://zhuanlan.zhihu.com/p/285918756
# @Link    : https://blog.csdn.net/jackytintin/article/details/79425866
# @Link    : https://github.com/DingKe/ml-tutorial/blob/master/ctc/CTC.ipynb
# @Link    : https://gist.github.com/awni/56369a90d03953e370f3964c826ed4b0

import numpy as np
from CTCutils import softmax, remove_blank, log_sum_exp, negative_infinity
from collections import defaultdict


# 前缀束搜索（Prefix Beam Search）
def prefix_beam_decode(log_probs, beam_size=2, blank=0):
    seq_len, embedding_size = log_probs.shape
    log_log_probs = np.log(log_probs)

    beam = [(tuple(), (0, negative_infinity))]  # blank, non-blank
    for i in range(seq_len):  # for every timestep
        new_beam = defaultdict(lambda: (negative_infinity, negative_infinity))

        for prefix, (p_b, p_nb) in beam:
            for j in range(embedding_size):  # for every state
                p = log_log_probs[i, j]

                if j == blank:  # propose a blank
                    new_p_b, new_p_nb = new_beam[prefix]
                    new_p_b = log_sum_exp(new_p_b, log_sum_exp(p_b + p, p_nb + p))
                    new_beam[prefix] = (new_p_b, new_p_nb)
                    continue
                else:  # extend with non-blank
                    end_t = prefix[-1] if prefix else None

                    # exntend current prefix
                    new_prefix = prefix + (j,)
                    new_p_b, new_p_nb = new_beam[new_prefix]
                    if j != end_t:
                        new_p_nb = log_sum_exp(new_p_nb, log_sum_exp(p_b + p, p_nb + p))
                    else:
                        new_p_nb = log_sum_exp(new_p_nb, p_b + p)
                    new_beam[new_prefix] = (new_p_b, new_p_nb)

                    # keep current prefix
                    if j == end_t:
                        new_p_b, new_p_nb = new_beam[prefix]
                        new_p_nb = log_sum_exp(new_p_nb, p_nb + p)
                        new_beam[prefix] = (new_p_b, new_p_nb)

        # top beam_size
        beam = sorted(new_beam.items(), key=lambda x: log_sum_exp(*x[1]), reverse=True)
        beam = beam[:beam_size]

    return beam


if __name__ == '__main__':
    net_out = np.random.random([20, 6])
    log_probs = softmax(net_out)
    beam = prefix_beam_decode(log_probs, beam_size=2)
    for string, score in beam:
        print(remove_blank(string), score)
