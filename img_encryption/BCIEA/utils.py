# -*-coding:utf-8 -*-

"""
# File       : utils
# Time       ：2024/10/6 18:34
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


# chaotic map
# x0:init value; h:control para; num:len of the seq
def PWLCM(x0, h, num):
    pwlcm = [0] * num
    pwlcm[0] = x0
    for i in range(1, num):
        if 0 < pwlcm[i - 1] < h:
            pwlcm[i] = pwlcm[i - 1] / h
        elif h <= pwlcm[i - 1] < 0.5:
            pwlcm[i] = (pwlcm[i - 1] - h) / (0.5 - h)
        elif 0.5 <= pwlcm[i - 1] < 1 - h:
            pwlcm[i] = (1 - pwlcm[i - 1] - h) / (0.5 - h)
        elif 1 - h <= pwlcm[i - 1] < 1:
            pwlcm[i] = (1 - pwlcm[i - 1]) / h
        else:
            raise ValueError("xi must be in [0, 1]")
    return pwlcm
