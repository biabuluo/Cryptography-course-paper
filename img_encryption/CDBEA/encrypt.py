# -*-coding:utf-8 -*-

"""
# File       : encrypt.py
# Time       ：2024/10/8 0:14
# Author     ：chenyu
# version    ：python 3.8
# Description：加密算法
"""
from math import floor
import cv2
import numpy as np
import os
from random import uniform
from copy import deepcopy

from CDBEA.utils import img_bit_decomposition, gen_keys
from CDBEA.chaos.PWLCW import PWLCM
from CDBEA.dna import sbn, permutation, DNA_encoding, DNA_decoding, DNA_operation


def encrypt(
        img_path,
        x0=None, h0=None, rounds=None, discard=None,  # keys
        params_path='params.npz',  # the path of saved params
        use_params=False,  # if to use used params
):
    # initialization
    if not use_params:
        x0 = uniform(1e-16, 1 - 1e-16) if not x0 else x0
        h0 = uniform(1e-16, 0.5 - 1e-16) if not h0 else h0
        rounds = 1 if not rounds else rounds
        discard = 1000 if not discard else discard
    else:
        use_params_path = f"./params/{params_path}"
        if os.path.exists(use_params_path):
            params = np.load(use_params_path)
            x0 = params['x0']
            h0 = params['h0']
            rounds = params['rounds']
            discard = params['discard']
        else:
            raise FileNotFoundError(f"can not find the params file: {use_params_path}")

    filename = img_path.split('/')[-1].split('.')
    filename, ext = filename[0], filename[-1]
    encrypt_img_path = f"{filename}_encrypt.png"
    # get img (grey img supported)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    H, W = img.shape
    length = 4 * 1 * H * W
    print(f"ENCRYPTION: img resolution is {H} * {W}")

    # get params
    keys = {
        'x0': x0,
        'h0': h0,
        'rounds': rounds,
        'discard': discard
    }
    x_list, h_list, d_list = gen_keys(keys)

    bitplanes = img_bit_decomposition(img)  # Image bit plane decomposition
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

    if not os.path.exists('./result'):
        os.mkdir('./result')
    # save img
    save_path = os.path.join('./result', encrypt_img_path)
    cv2.imwrite(save_path, res)

    # save params
    if not use_params:
        if not os.path.exists('./params'):
            os.mkdir('./params')
        np.savez(f'./params/{params_path}', x0=x0, h0=h0, rounds=rounds, discard=discard)
    print("ENCRYPTION DONE!")
    return res, save_path


if __name__ == '__main__':
    res, _ = encrypt('../images/grey/mandril.jpg', use_params=False)
    print(res.shape)
