# -*-coding:utf-8 -*-

"""
# File       : histogram.py
# Time       ：2024/10/6 23:27
# Author     ：chenyu
# version    ：python 3.8
# Description：
"""
import warnings
warnings.filterwarnings('ignore')
import cv2
from matplotlib import pyplot as plt


# 生成直方图
def histogram(origin_p, encrypt_p, save_p):
    from PIL import Image  # 如果你需要从文件中加载图像
    import numpy as np
    # 方法1：直接从文件加载图像（使用PIL库）
    img1 = Image.open(origin_p)
    img1 = np.array(img1)
    img2 = Image.open(encrypt_p)
    img2 = np.array(img2)

    src1 = cv2.imread(origin_p, cv2.IMREAD_GRAYSCALE)
    src2 = cv2.imread(encrypt_p, cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(12, 3))
    plt.title("histogram")
    plt.subplot(141)
    plt.imshow(img1, cmap='gray')
    plt.subplot(142)
    plt.hist(src1.ravel(), 256)
    plt.subplot(143)
    plt.imshow(img2, cmap='gray')
    plt.subplot(144)
    plt.hist(src2.ravel(), 256)
    plt.tight_layout()
    # plt.savefig(save_p)
    plt.show()

if __name__ == '__main__':
    histogram('../images/grey/med.png', '../CDBEA/result/med_encrypt.png', None)
