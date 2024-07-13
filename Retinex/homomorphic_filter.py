import cv2
import numpy as np


def homomorphic_filter(src, d0=10, rl=0.5, rh=2.0, c=4, h=2.0, l=0.5):
    gray = src.copy()
    if len(src.shape) > 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # 转换成灰度图像

    gray = np.log(1e-5 + gray)  # 取对数

    rows, cols = gray.shape
    gray_fft = np.fft.fft2(gray)  # FFT傅里叶变换
    gray_fftshift = np.fft.fftshift(gray_fft)  # FFT中心化

    M, N = np.meshgrid(
        np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2)
    )
    D = np.sqrt(M**2 + N**2)  # 计算距离
    Z = (rh - rl) * (1 - np.exp(-c * (D**2 / d0**2))) + rl  # H(u,v)传输函数

    dst_fftshift = Z * gray_fftshift
    dst_fftshift = (h - l) * dst_fftshift + l
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)

    dst_ifft = np.fft.ifft2(dst_ifftshift)  # IFFT逆傅里叶变换
    dst = np.real(dst_ifft)  # IFFT取实部

    dst = np.exp(dst) - 1  # 还原
    dst = np.uint8(np.clip(dst, 0, 255))
    return dst


img = cv2.imread(
    "E:\\ExposureDetect\\imgs2\\192.168.40.151_01_20240116134335477_1_11.jpg"
)
img_new = homomorphic_filter(img)

cv2.imshow("img", img)
cv2.imshow("img_new", img_new)
cv2.imwrite("img_new1.jpg", img_new)

key = cv2.waitKey(0)
cv2.destroyAllWindows()
