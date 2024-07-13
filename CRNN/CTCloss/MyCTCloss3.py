# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/29 18:38
# @Author  : Jis-Baos
# @File    : MyCTCloss3.py

import numpy as np

# state
seq_len = 20
target_len = 5

# mapping tables
mapping_tables = ['s', 't', 'a', 't', 'e']

# the column of cal_map
cal_map_column = ['-', 's', '-', 't', '-', 'a', '-', 't', '-', 'e', '-']

cal_map = np.full(shape=(2 * target_len + 1, seq_len), fill_value=1 / (2 * target_len + 1), dtype=np.float64)
cal_map = cal_map.transpose(1, 0)
print(cal_map.shape)
cal_map_accumulate = np.zeros_like(cal_map, dtype=np.float64)

# 初始化动态规划的初始时刻————时刻0
cal_map_accumulate[0][0] = cal_map[0][0]
cal_map_accumulate[0][1] = cal_map[0][1]

# if i % 2 == 0:
#     alpha = '-'
#     print(alpha)
# else:
#     alpha = mapping_tables[int(i / 2)]
#     print(alpha)

# 从时刻1开始遍历
for i in range(1, seq_len - 1):
    print("seq_len: ", i)
    # 实际上，这里最大的遍历长度仅是： 2 * (i + 1)
    for j in range(2 * (i + 1)):
        if j <= 2 * target_len:
            print("alpha: ", j)
            # 先处理blank
            if j % 2 == 0 and j == 0:
                cal_map_accumulate[i][j] = cal_map[i][j] * cal_map_accumulate[i - 1][j]
            elif j % 2 == 0:
                if j == 2 * i and j < 2 * target_len:
                    cal_map_accumulate[i][j] = cal_map[i][j] * cal_map_accumulate[i - 1][j - 1]
                elif j == 2 * target_len and i >= target_len:
                    cal_map_accumulate[i][j] = cal_map[i][j] * cal_map_accumulate[i - 1][j]
            elif j % 2 == 0 and 0 < j < 2 * i and j < 2 * target_len:
                cal_map_accumulate[i][j] = cal_map[i][j] * (
                            cal_map_accumulate[i - 1][j] + cal_map_accumulate[i - 1][j - 1])

            # 处理字符
            elif j // 2 == 1 and j == 1:
                cal_map_accumulate[i][j] = cal_map[i][j] * (
                            cal_map_accumulate[i - 1][j] + cal_map_accumulate[i - 1][j - 1])
            elif j // 2 == 1:
                if j == 2 * i + 1 and j < 2 * target_len:
                    cal_map_accumulate[i][j] = cal_map[i][j] * cal_map_accumulate[i - 1][j - 2]
                elif j == 2 * target_len - 1 and i >= target_len:
                    cal_map_accumulate[i][j] = cal_map[i][j] * cal_map_accumulate[i - 1][j - 1]
            elif j // 2 == 1 and 1 < j < 2 * i + 1 and j < 2 * target_len:
                cal_map_accumulate[i][j] = cal_map[i][j] * (
                        cal_map_accumulate[i - 1][j] + cal_map_accumulate[i - 1][j - 1] + cal_map_accumulate[i - 1][
                    j - 2])

print(cal_map)
print(cal_map_accumulate)
print("hello world!")
