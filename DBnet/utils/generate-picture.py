# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/12 21:40
# @Author  : Jis-Baos
# @File    : generate-picture.py

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ttf_path = "D:/PythonProject/DBnet_pytorch/utils/aramisi.ttf"

save_path = "D://PythonProject//DBnet_pytorch//utils//test.jpg"
text_size = 50  # text_size表示字号
font = ImageFont.truetype(ttf_path, text_size)  # 返回一个字体对象

# create a blank cancas with extra space between lines
text_width, text_height = font.getsize('hello')

canvas = Image.new('RGB', [224, 224], (255, 255, 255))

# draw the text onto the canvas
draw = ImageDraw.Draw(canvas)
white = "#000000"
coord_x = np.random.randint(0, 224 - text_width)
coord_y = np.random.randint(0, 224 - text_height)

print(coord_x, coord_y)
print(coord_x + text_width, coord_y)
print(coord_x + text_width, coord_y + text_height)
print(coord_x, coord_y + text_height)

draw.text((coord_x, coord_y), 'hello', font=font, fill=white)
canvas.show()
canvas.save(save_path, quality=95)
