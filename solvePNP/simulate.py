import numpy as np
import cv2 as cv
import math
import random
from math import degrees as dg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def aaa(a, sigma):
    objp = np.zeros((10 * 10, 3), np.float32)
    objp[:, :2] = np.mgrid[0:200:20, 0:200:20].T.reshape(-1, 2)

    f = 8
    dx = 0.01
    dy = 0.01
    u0 = 320
    v0 = 240
    list1 = [f / dx, 0, u0, 0, 0, f / dy, v0, 0, 0, 0, 1, 0]
    M1 = np.mat(list1).reshape(3, 4)
    #########################
    in_site = np.mat([a, 0, 0])
    in_rr = in_site / 180 * math.pi
    in_r = cv.Rodrigues(in_rr, jacobian=0)[0]

    in_weiyi = np.mat([0, 0, 1000])

    M2 = np.hstack((in_r, in_weiyi.T))
    yi = np.mat([0, 0, 0, 1])
    M2 = np.vstack((M2, yi))

    M = M1 * M2
    ###############################################################
    corner_out = np.zeros((10 * 10, 2), np.float32)
    k = 0
    for l in objp:

        l = np.append(l, 1)
        l = np.mat(l)

        out = (M1 * M2 * l.T) / ((M2 * l.T)[2, 0])

        corner_out[k][0] = float(out[0, 0]) + random.gauss(0, sigma)
        corner_out[k][1] = float(out[1, 0]) + random.gauss(0, sigma)
        k += 1
    list1 = [f / dx, 0, u0, 0, f / dy, v0, 0, 0, 1]

    mtx = np.mat(list1).reshape(3, 3)
    dist = np.mat([0.0, 0.0, 0.0, 0.0, 0.0])
    _, R, T = cv.solvePnP(objp, corner_out, mtx, dist)
    sita_x = dg(R[0][0])
    sita_y = dg(R[1][0])
    sita_z = dg(R[2][0])

    return abs(sita_x) - abs(a)


if __name__ == "__main__":
    ###########################################
    aa = []
    b = []
    for a in range(0, 90, 1):
        print("只绕x", a, "角度时的角度标定误差", aaa(a, 0.1))
        b.append(a)
        aa.append(aaa(a, 0.1))

    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(0, 90, 1)  # 角度
    Y = np.arange(0, 100, 1)  # 次数

    a = []
    for i in range(0, len(Y)):
        for j in range(0, len(X)):
            z = aaa(X[j], Y[i])
            a.append(z)

            # print(X[j], Y[i])
            # print(z)

    Z = np.mat(a).reshape(len(Y), len(X))
    X, Y = np.meshgrid(X, Y)

    # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")
    plt.savefig("test.png", dpi=300)
    # plt.show()
