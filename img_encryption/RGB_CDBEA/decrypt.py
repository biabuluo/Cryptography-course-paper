# -*-coding:utf-8 -*-

"""
# File       : decryption
# Time       ：2024/10/8 22:15
# Author     ：chenyu
# version    ：python 3.8
# Description：
"""
from math import floor
import cv2
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from utils import gen_keys, img_bit_decomposition
from chaos.PWLCW import PWLCM
from dna import sbn, permutation, permutation_reverse, DNA_decoding, DNA_encoding, DNA_operation


def decrypt(matrix, keys):
    # print(keys)
    # load key
    x0, h0, rounds, discard = keys['x0'], keys['h0'], keys['rounds'], keys['discard']

    H, W = matrix.shape  # M: height, N: width
    print(f"DECRYPTION: matrix resolution is {H} * {W}")

    x_list, h_list, d_list = gen_keys(keys)
    # reverse
    x_list, h_list, d_list = x_list[::-1], h_list[::-1], d_list[::-1]

    bitplanes = img_bit_decomposition(matrix)  # img discomposition
    a1 = bitplanes[:4, :, :].ravel()
    a2 = bitplanes[4:, :, :].ravel()

    # start decryption
    length = H * W * 4
    for i in range(rounds):
        # gen keys_round
        k = np.array(PWLCM(x_list[i], h_list[i], 2 * length + d_list[i])[d_list[i]:])
        k1, k2 = k[::2], k[1::2]
        k11 = np.round(k1 * 1e8).astype(np.uint8) % 2
        k22 = np.round(k2 * 1e8).astype(np.uint8) % 2
        # decide which rule to use
        rule_num = (sum(k11) + sum(k22)) % 8
        k33 = np.bitwise_xor(k11, k22)

        # diffusion
        k_op = np.round(k2 * 1e8).astype(np.uint8) % 7
        # 替换5和6 使用布尔索引进行替换
        k_op[k_op == 5] = 7  # 临时替换 5 为 7
        k_op[k_op == 6] = 5  # 将 6 替换为 5
        k_op[k_op == 7] = 6  # 将临时的 7 替换为 6
        a_dna = DNA_encoding(a1, a2, rule_num)
        k_dna = DNA_encoding(k11, k22, rule_num)
        a_new_dna = DNA_operation(a_dna, k_dna, k_op)
        a1, a2 = DNA_decoding(a_new_dna, rule_num)

        # confusion:permutation
        # permutation
        k1_index = np.argsort(k1)
        a1, a2 = list(a1), list(a2)
        a1, a2 = permutation_reverse(a1, a2, k1_index)
        # 单碱基突变
        a1, a2 = sbn(a1, a2, k33)

    # A1，A2两个序列合并成原始图像
    A = np.append(a1, a2)[::-1].reshape(8, H, W)[:, ::-1, ::-1]
    res = np.zeros((H, W), dtype=np.uint8)
    for i in range(8):
        res = cv2.bitwise_or(res, A[i, :, :] << i)

    return res
