# 此文件用于处理ICDAR2015数据集

import os
from shutil import move


def sort_files(directory_path):
    for filename in os.listdir(directory_path):
        print("file's name: {}".format(filename))
        if os.path.isfile(os.path.join(directory_path, filename)):
            # 获取文件扩展名
            file_extension = filename.split(".")[-1]
            # 创建目标目录
            destination_directory = os.path.join(directory_path, file_extension)
            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)
            # 移动文件
            move(
                os.path.join(directory_path, filename),
                os.path.join(destination_directory, filename),
            )


def obtain_label(file_path, save_path):
    with open(file_path, "r") as file_r:
        for line in file_r.readlines():
            name = line.split(".")[0] + ".txt"
            label = line.split(",")[-1]
            label = label.replace('"', '')
            label = label.replace(' ', '')
            path = os.path.join(save_path, name)

            print(path, label)
            with open(path, "w") as file_w:
                file_w.write(label)
                file_w.close()
            


if __name__ == "__main__":
    # 调用函数，替换路径
    # directory_path = "E:/OCRData/ch4_training_word_images_gt"
    # sort_files(directory_path)

    # file_path = "E:/OCRData/ch4_training_word_images_gt/txt/gt.txt"
    # save_path = "E:/OCRData/ch4_training_word_images_gt/label"
    # obtain_label(file_path, save_path)
    
    directory_path = "E:/OCRData/ch4_training_word_images_gt/data"
    for filename in os.listdir(directory_path):
        print("file's name: {}".format(filename.split(".")[0]))

    print("Done...")
