# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/30 18:48
# @Author  : Jis-Baos
# @File    : CTCtest.py

import numpy as np
from CTCutils import softmax
from CTCloss1 import forward, backward, gradient
from CTCloss2 import forward_log, backward_log, gradient_log


print("**********************************init the log_probs**********************************")
# seq_len, embedding_size
net_input = np.random.random([12, 5])
log_probs = softmax(net_input)
print(log_probs)
# 验证softmax是否成立
print(log_probs.sum(1, keepdims=True))

labels_padded = [0, 3, 0, 3, 0, 4, 0]  # 0 for blank

log_alpha1 = forward(log_probs, labels_padded)
log_beta1 = backward(log_probs, labels_padded)
log_probs_grad1 = gradient(log_probs, labels_padded)

log_alpha2 = forward_log(np.log(log_probs), labels_padded)
log_beta2 = backward_log(np.log(log_probs), labels_padded)
log_probs_grad2 = gradient_log(np.log(log_probs), labels_padded)


print("log_alpha1： ", log_alpha1)
print("log_alpha2： ", np.exp(log_alpha2))
print("log_beta1： ", log_beta1)
print("log_beta2： ", np.exp(log_beta2))
print("log_probs_grad1： ", log_probs_grad1)
print("log_probs_grad2： ", np.exp(log_probs_grad2))