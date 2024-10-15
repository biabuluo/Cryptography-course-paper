# -*-coding:utf-8 -*-

"""
# File       : utils
# Time       ：2024/10/8 0:15
# Author     ：chenyu
# version    ：python 3.8
# Description：
"""
import cv2
from numpy import uint8, zeros
from math import floor

from chaos.PWLCW import PWLCM

# Image bit plane decomposition
def img_bit_decomposition(img):
    m, n = img.shape
    r = zeros((8, m, n), dtype=uint8)
    for i in range(8):
        r[i, :, :] = cv2.bitwise_and(img, 2 ** i)
        mask = r[i, :, :] > 0
        r[i, mask] = 1
    return r

def gen_keys(keys):
    x0, h0, rounds, discard = keys['x0'], keys['h0'], keys['rounds'], keys['discard']
    # use chaos map to gen key_gen
    x_list = []
    h_list = []
    d_list = []
    seq = PWLCM(x0, h0, discard + rounds * 3)[discard:]
    for i in range(rounds):
        x_list.append(seq[i * 3])
        h_list.append(seq[i * 3 + 1] / 2)
        d_list.append(floor(seq[i * 3 + 2] * 1e14) % 900 + 100)
    return x_list, h_list, d_list