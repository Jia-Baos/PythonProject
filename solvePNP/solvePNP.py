from math import degrees as dg
import numpy as np
import cv2 as cv
import math
import glob
import random


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotation2Euler(R):

    assert isRotationMatrix(R)

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


if __name__ == "__main__":

    # 加载相机标定的内参数、畸变参数
    f = 8
    dx = 0.01
    dy = 0.01
    u0 = 320
    v0 = 240
    list1 = [f / dx, 0, u0, 0, f / dy, v0, 0, 0, 1]

    mtx = np.mat(list1).reshape(3, 3)
    dist = np.mat([0, 0, 0, 0, 0])

    # 世界坐标系下的物体位置矩阵（Z=0）
    objp = np.zeros((10 * 10, 3), np.float32)
    objp[:, :2] = np.mgrid[0:200:20, 0:200:20].T.reshape(-1, 2)

    # 读取图片
    test_img = cv.imread("output.jpg")
    gray = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)

    # 找到图像平面点角点坐标
    corners = np.load("corner.npy")
    ret = True

    if ret:
        _, R, T = cv.solvePnP(objp, corners, mtx, dist)
        cv.solvePnPRefineLM(objp, corners, mtx, dist, R, T)
        print("所求结果：")
        print("旋转向量", R, R.shape)
        print("平移向量", T)

        sita_x = dg(R[0][0])
        sita_y = dg(R[1][0])
        sita_z = dg(R[2][0])
        print("sita_x is  ", sita_x)
        print("sita_y is  ", sita_y)
        print("sita_z is  ", sita_z)
        # print("rotation2Euler: ", rotation2Euler(R))
