import numpy as np
import cv2 as cv
import math
import random

if __name__ == "__main__":
    ###############世界坐标系的3d点###############
    objp = np.zeros((10 * 10, 3), np.float32)
    objp[:, :2] = np.mgrid[0:200:20, 0:200:20].T.reshape(-1, 2)
    print("objp shape: {}".format(objp.shape))

    ################相机内参数矩阵################
    f = 8
    dx = 0.01
    dy = 0.01
    u0 = 320
    v0 = 240
    list1 = [f / dx, 0, u0, 0, 0, f / dy, v0, 0, 0, 0, 1, 0]
    M1 = np.mat(list1).reshape(3, 4)
    print("M1: {}".format(M1))
    #####################相机外参数矩阵#################

    # 输入旋转向量，并转化为旋转矩阵
    # x = input("请输入旋转向量_类四元数，使用逗号隔开: ")
    # x = "0,0,0"
    # x = "0,0,45"
    # x = "0,45,0"
    # x = "80,0,0"
    x = "45,45,0"
    list1 = x.split(",")
    list1 = [float(list1[i]) for i in range(len(list1))]
    in_site = np.mat(list1)
    in_rr = in_site / 180 * math.pi

    # 旋转向量转化为旋转矩阵
    in_r = cv.Rodrigues(in_rr, jacobian=0)[0]

    # 输入平移向量
    # y = input("请输入平移向量，使用逗号隔开: ")
    y = "0,0,1000"
    list2 = y.split(",")
    list2 = [float(list2[i]) for i in range(len(list2))]

    in_weiyi = np.mat(list2)

    # 获得外参数矩阵
    # 列合并
    M2 = np.hstack((in_r, in_weiyi.T))
    print("M2: {}".format(M2))
    yi = np.mat([0, 0, 0, 1])
    M2 = np.vstack((M2, yi))
    print("M2: {}".format(M2))

    ########################相机矩阵###################################
    M = M1 * M2
    print("M: {}".format(M))
    ########################创建空白图像############################################
    img = np.zeros((240 * 2, 320 * 2, 3), np.uint8)
    img.fill(255)

    ##########对每个点进行透视运算，将世界坐标转换为像素坐标，并将其标记在空白图像中##############
    corner_out = np.zeros((10 * 10, 2), np.float32)

    k = 0
    sigma = 0.12
    for l in objp:
        print("3d points: {}".format(l))
        l = np.append(l, 1)
        l = np.mat(l)
        print(l)

        out = (M1 * M2 * l.T) / ((M2 * l.T)[2, 0])
        corner_out[k][0] = float(out[0, 0])
        corner_out[k][1] = float(out[1, 0])

        # sigma为高斯噪声的尺度，0为均值
        # corner_out[k][0] = float(out[0, 0]) + random.gauss(0, sigma)
        # corner_out[k][1] = float(out[1, 0]) + random.gauss(0, sigma)

        img = cv.circle(
            img, (int(corner_out[k][0]), int(corner_out[k][1])), 1, (0, 0, 255), 4
        )
        k += 1

    np.save("corner.npy", corner_out)
    cv.imwrite("output.jpg", img)
    cv.imshow("image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
