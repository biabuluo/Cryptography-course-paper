# -*-coding:utf-8 -*-

"""
# File       : experiment
# Time       ：2024/10/7 16:41
# Author     ：chenyu
# version    ：python 3.8
# Description：模拟加密解密
"""
import numpy as np
from random import uniform
from CDBEA.chaos.PWLCW import PWLCM
from math import floor
from copy import deepcopy

from CDBEA.dna import DNA_encoding, DNA_decoding, DNA_operation, sbn, permutation, permutation_reverse
import time

# start = time.time()
# C = 1
# H = 512
# W = 512


def get_distence(a, b):
    return np.sum(np.abs(a - b))



# a1 = np.random.randint(2, size=length)
# a2 = np.random.randint(2, size=length)




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

def encrypt(A1, A2, C=1, H=64, W=64):
    length = 4 * C * H * W
    keys = {
        # 'x0': uniform(1e-16, 1 - 1e-16),
        # 'h0': uniform(1e-16, 0.5 - 1e-16),
        'x0': 0.6952020325739682,
        'h0': 0.06975026107356343,
        'rounds': 1,
        'discard': 1000
    }
    x0, h0, rounds, discard = keys['x0'], keys['h0'], keys['rounds'], keys['discard']
    x_list, h_list, d_list = gen_keys(keys)
    a1, a2 = deepcopy(A1), deepcopy(A2)
    for i in range(rounds):
        # gen keys_round
        k = np.array(PWLCM(x_list[i], h_list[i], 2 * length + d_list[i])[d_list[i]:])
        k1, k2 = k[::2], k[1::2]
        k11 = np.round(k1*1e8).astype(int)%2
        k22 = np.round(k2*1e8).astype(int)%2

        # decide which rule to use
        rule_num = (sum(k11) + sum(k22)) % 8
        k33 = np.bitwise_xor(k11, k22)

        # confusion:permutation
        # 单碱基突变
        a1, a2 = sbn(a1, a2, k33)
        # permutation
        k1_index = np.argsort(k1)
        a1, a2 = permutation(a1, a2, k1_index)
        # diffusion
        k_op = np.round(k2*1e8).astype(int) % 7

        a_dna = DNA_encoding(a1, a2, rule_num)
        k_dna = DNA_encoding(k11, k22, rule_num)
        a_new_dna = DNA_operation(a_dna, k_dna, k_op)
        a1, a2 = DNA_decoding(a_new_dna, rule_num)
    return a1, a2

def decrypt(keys, A1, A2, C=1, H=64, W=64):
    length = 4 * C * H * W
    x0, h0, rounds, discard = keys['x0'], keys['h0'], keys['rounds'], keys['discard']
    x_list, h_list, d_list = gen_keys(keys)
    # reverse
    x_list, h_list, d_list = x_list[::-1], h_list[::-1], d_list[::-1]
    a1, a2 = deepcopy(A1), deepcopy(A2)
    for i in range(rounds):
        # gen keys_round
        k = np.array(PWLCM(x_list[i], h_list[i], 2 * length + d_list[i])[d_list[i]:])
        k1, k2 = k[::2], k[1::2]
        k11 = np.round(k1*1e8).astype(int)%2
        k22 = np.round(k2*1e8).astype(int)%2
        # decide which rule to use
        rule_num = (sum(k11) + sum(k22)) % 8
        k33 = np.bitwise_xor(k11, k22)

        # diffusion
        k_op = np.round(k2*1e8).astype(int) % 7
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

    return a1, a2


# a1_new, a2_new = encrypt(keys, a1, a2)
# print(get_distence(a1, a1_new))
# print(get_distence(a2, a2_new))
#
# a1_new, a2_new = decrypt(keys, a1_new, a2_new)
# print(get_distence(a1, a1_new))
# print(get_distence(a2, a2_new))
#
# end = time.time()
# print(end-start)