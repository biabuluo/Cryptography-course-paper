# -*-coding:utf-8 -*-

"""
# File       : encrypt.py
# Time       ：2024/10/8 0:14
# Author     ：chenyu
# version    ：python 3.8
# Description：加密算法
"""
import cv2
import numpy as np

from utils import img_bit_decomposition, gen_keys
from chaos.PWLCW import PWLCM
from dna import sbn, permutation, DNA_encoding, DNA_decoding, DNA_operation


def encrypt(
        matrix,
        keys
):
    H, W = matrix.shape
    length = 4 * 1 * H * W
    print(f"ENCRYPTION: matrix resolution is {H} * {W}")
    rounds = keys['rounds']
    x_list, h_list, d_list = gen_keys(keys)

    bitplanes = img_bit_decomposition(matrix)  # Image bit plane decomposition
    bitplanes = bitplanes[::-1, :, :]  # reverse
    a1 = bitplanes[:4, :, :].ravel()  # high bits
    a2 = bitplanes[4:, :, :].ravel()  # low bits

    # start to encryption
    for i in range(rounds):
        # gen keys_round
        k = np.array(PWLCM(x_list[i], h_list[i], 2 * length + d_list[i])[d_list[i]:])
        k1, k2 = k[::2], k[1::2]
        k11 = np.round(k1 * 1e8).astype(np.uint8) % 2
        k22 = np.round(k2 * 1e8).astype(np.uint8) % 2
        # decide which rule to use
        rule_num = (sum(k11) + sum(k22)) % 8
        k33 = np.bitwise_xor(k11, k22)
        # confusion:sbn+permutation
        # 单碱基突变
        a1, a2 = sbn(a1, a2, k33)

        # permutation
        k1_index = np.argsort(k1)
        a1, a2 = permutation(a1, a2, k1_index)

        # diffusion:dna operation
        k_op = np.round(k2 * 1e8).astype(np.uint8) % 7

        a_dna = DNA_encoding(a1, a2, rule_num)
        k_dna = DNA_encoding(k11, k22, rule_num)
        a_new_dna = DNA_operation(a_dna, k_dna, k_op)
        a1, a2 = DNA_decoding(a_new_dna, rule_num)

    # composition the encrypted img
    C = np.append(a1, a2).reshape(8, H, W)
    res = np.zeros((H, W), dtype=np.uint8)
    for i in range(8):
        tmp = C[i, :, :] << i
        res = cv2.bitwise_or(res, tmp)

    return res
