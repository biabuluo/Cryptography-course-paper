# -*-coding:utf-8 -*-

"""
# File       : sensity
# Time       ：2024/10/14 23:09
# Author     ：chenyu
# version    ：python 3.8
# Description：
"""

import cv2
from matplotlib import pyplot as plt
from CDBEA.encrypt import encrypt
from CDBEA.decrypt import decrypt
import numpy as np
from copy import deepcopy


def encrypt_sensitivity(img_path, key=None, modified_key=None):
    key = [0.01234567890123, 0.12345678912345] if not key else key
    modified_key = [0.01234567890124, 0.12345678912345] if not modified_key else modified_key

    # 结果展示
    # 原图像
    r_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
    plt.subplot(141)
    plt.imshow(r_img, cmap='gray')
    plt.title('origin')

    # key加密的图像
    key_img, _ = encrypt(img_path, *key)
    plt.subplot(142)
    plt.imshow(key_img, cmap='gray')
    plt.title('correct encryption')

    # 修改过的key加密的图像
    modified_key_img, _ = encrypt(img_path, *modified_key)
    plt.subplot(143)
    plt.imshow(modified_key_img, cmap='gray')
    plt.title('incorrect encryption')

    # 两者差值
    plt.subplot(144)
    plt.imshow(np.abs(modified_key_img - key_img), cmap='gray')
    plt.title('difference')
    plt.tight_layout()
    plt.show()


def decrypt_sensitivity(img_path, raw_img_path, key=None, incorrect_key=None):
    key = [0.01234567890123, 0.12345678912345] if not key else key
    if not incorrect_key:
        incorrect_key = deepcopy(key)
        incorrect_key[0] += 0.00000000000001

    # 结果展示
    # 原图像
    r_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
    plt.subplot(141)
    plt.imshow(r_img, cmap='gray')
    plt.title('origin')

    # 原始key加密的图像
    key_img, encrypt_img_path = encrypt(img_path, *key)
    plt.subplot(142)
    plt.imshow(key_img, cmap='gray')
    plt.title('cipher')

    # 修改过的key解密的图像
    # 读取参数
    raw_params = np.load('./params/params.npz')
    np.savez('./params/incorrect_params.npz',
             x0=incorrect_key[0], h0=incorrect_key[1],
             rounds=1, discard=1000)
    # print(encrypt_img_path)
    incorrect_decrypt_img = decrypt(
        encrypt_img_path, raw_img_path, params_path='incorrect_params.npz')
    plt.subplot(143)
    plt.imshow(incorrect_decrypt_img, cmap='gray')
    plt.title('incorrect decryption')

    # 原始key解密的图像
    decrypt_img = decrypt(encrypt_img_path,raw_img_path,  params_path='params.npz')
    plt.subplot(144)
    plt.imshow(decrypt_img, cmap='gray')
    plt.title('correct decryption')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img_path = "../images/grey/cameraman.jpg"
    raw_img_path = "../images/grey/cameraman.jpg"
    encrypt_sensitivity(img_path)
    decrypt_sensitivity(img_path, raw_img_path, None, None)
