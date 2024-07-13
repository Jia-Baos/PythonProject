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


if __name__ == "__main__":
    # 调用函数，替换路径
    directory_path = "D:/PythonProject/Test"
    sort_files(directory_path)
    print("Done...")
