# -*-coding:utf-8 -*-

"""
# File       : utils
# Time       ：2024/10/14 15:57
# Author     ：chenyu
# version    ：python 3.8
# Description：
"""
import cv2
from numpy import uint8, zeros

# Image bit plane decomposition
def img_bit_decomposition(img):
    m, n = img.shape
    r = zeros((8, m, n), dtype=uint8)
    for i in range(8):
        r[i, :, :] = cv2.bitwise_and(img, 2 ** i)
        mask = r[i, :, :] > 0
        r[i, mask] = 1
    return r

