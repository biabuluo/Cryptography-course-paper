# -*-coding:utf-8 -*-

"""
# File       : entropy
# Time       ：2024/10/7 0:20
# Author     ：chenyu
# version    ：python 3.8
# Description：信息熵
"""
import cv2
import math
import numpy as np

def _entropy(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    gray, num = np.unique(img, return_counts=True)
    entropy = 0

    for i in range(len(gray)):
        p = num[i]/(w*h)
        entropy -= p*(math.log(p, 2))
    return entropy


def entropy(raw_img, encrypt_img):
    # 图像lena的熵
    raw_entropy = _entropy(raw_img)
    encrypt_entropy = _entropy(encrypt_img)
    print('==================信息熵==================')
    print('原图像: \t{:.5}'.format(raw_entropy))
    print('加密图像: \t{:.5}'.format(encrypt_entropy))

if __name__ == '__main__':
    entropy('../images/grey/cameraman.jpg', '../CDBEA/result/cameraman_encrypt.png')
    entropy('../images/grey/med.png', '../CDBEA/result/med_encrypt.png')
    entropy('../images/grey/blonde.jpg', '../CDBEA/result/blonde_encrypt.png')
    entropy('../images/grey/darkhair.jpg', '../CDBEA/result/darkhair_encrypt.png')
    entropy('../images/grey/livingroom.jpg', '../CDBEA/result/livingroom_encrypt.png')
    entropy('../images/grey/mandril.jpg', '../CDBEA/result/mandril_encrypt.png')
