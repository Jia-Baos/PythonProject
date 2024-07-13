import os
import numpy as np


# 加载字符词典
dict_path = "D:/PythonProject/CRNN/src/dict.txt"

with open(dict_path, "r") as file_r:
    lines = file_r.readlines()

CHAR_LIST = [line.strip() for line in lines]

NUM_CLASSES = len(CHAR_LIST) + 1  # from dict.txt，字符个数+空白符

MAX_LENGTH = 20 # 数据集中label的最大长度为20

def encode(text):
    """
    返回text中每个字符在dict中的索引

    text: 原始字符串
    """
    indices = np.array([CHAR_LIST.index(c) for c in text])
    return indices


def decode(indices, length):
    """
    根据索引从dict中恢复原始字符串

    indices: 字符串索引
    length: 字符串真实长度
    """

    text = [CHAR_LIST[i] for i in indices]
    return text[:length]


if __name__ == "__main__":
    
    text = "0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~!\"#$%&'()*+,-./"
    indices = encode(text)
    print(indices)

    text = decode(indices, len(text))
    print(text)
    print(len(text))
