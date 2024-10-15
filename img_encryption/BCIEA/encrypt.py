# -*-coding:utf-8 -*-

"""
# File       : encrypt
# Time       ：2024/10/6 18:35
# Author     ：chenyu
# version    ：python 3.8
# Description：
"""
from math import floor
import cv2
import numpy as np
import os
from random import uniform
from copy import deepcopy

from BCIEA.utils import img_bit_decomposition, PWLCM

def encrypt(
        img_path,
        x0=None, u1=None, y0=None, u2=None,  # keys
        n_round=1,  # epochs
        params_path='params.npz',  # the path of saved params
        use_params=False,  # if to use used params
):
    # initialization
    if not use_params:
        x0 = uniform(1e-16, 1 - 1e-16) if not x0 else x0
        u1 = uniform(1e-16, 0.5 - 1e-16) if not u1 else u1
        y0 = uniform(1e-16, 1 - 1e-16) if not y0 else y0
        u2 = uniform(1e-16, 0.5 - 1e-16) if not u2 else u2
    else:
        use_params_path = f"./params/{params_path}"
        if os.path.exists(use_params_path):
            params = np.load(use_params_path)
            x0 = params['x0']
            u1 = params['u1']
            y0 = params['y0']
            u2 = params['u2']
            n_round = params['n_round']
        else:
            raise FileNotFoundError(f"can not find the params file: {use_params_path}")

    filename = img_path.split('/')[-1].split('.')
    filename, ext = filename[0], filename[-1]
    encrypt_img_path = f"{filename}_encrypt.png"
    # get img (grey img supported)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    M, N = img.shape  # M: height, N: width
    print(f"ENCRYPTION: img resolution is {N} * {M}")

    bitplanes = img_bit_decomposition(img)  # Image bit plane decomposition
    bitplanes = bitplanes[::-1, :, :]  # reverse
    A1 = bitplanes[:4, :, :].ravel()  # high bits
    A2 = bitplanes[4:, :, :].ravel()  # low bits

    # gen key1
    # Piecewise Logistic Chaotic Map
    N0 = 1000  # chaotic map discard number
    PWLCM_MAP_X = PWLCM(x0, u1, N0 + M * N)[N0:]

    X1 = [floor(i * 1e14) % 256 for i in PWLCM_MAP_X]  # map to 0-255
    X1_reshape = np.matrix(X1, dtype=np.uint8).reshape(M, N)
    # print(X1_reshape)
    # print(type(X1_reshape))
    X1_bitplanes = img_bit_decomposition(X1_reshape)
    b1 = X1_bitplanes[::2, :, :].ravel()  # key1 even bits
    b2 = X1_bitplanes[1::2, :, :].ravel()  # key1 odd bits

    # start encryption rounds
    L = M * N * 4
    A11_0 = []
    A22_0 = []
    for k in range(n_round):
        # initial next status of A1 A2 after one round diffusion and confusion
        B1 = np.zeros(L, dtype=np.uint8)
        B2 = np.zeros(L, dtype=np.uint8)

        # diffusion
        sum1 = np.sum(A2)
        # print(A1.shape, sum1.shape)
        A11 = np.roll(A1, sum1)
        for i in range(L):
            B1[i] = A11[i] ^ A11[i - 1] ^ A2[i] ^ b1[i]
        sum2 = np.sum(B1)
        A22 = np.roll(A2, sum2)
        for i in range(L):
            B2[i] = A22[i] ^ A22[i - 1] ^ B1[i] ^ b2[i]

        # confusion
        # gen Y and Z
        sum = np.sum(B1) + np.sum(B2)
        s0 = (y0 + sum / L) % 1  # key2 init value s0
        S = PWLCM(s0, u2, N0 + 2 * L)[N0:]
        S1, S2 = S[:L], S[L:]
        Y = [floor(s1 * 1e14) % L for s1 in S1]  # map to [0-L-1]
        Z = [floor(s2 * 1e14) % L for s2 in S2]

        for i in range(L):
            B1[i], B2[Y[i]] = B2[Y[i]], B1[i]
        for j in range(L):
            B2[j], B1[Z[j]] = B1[Z[j]], B2[j]
        A11_0.append(A11[0])
        A22_0.append(A22[0])
        # update A1 A2
        A1 = deepcopy(B1)
        A2 = deepcopy(B2)

    # composition the encrypted img
    C = np.append(B1, B2).reshape(8, M, N)
    res = np.zeros((M, N), dtype=np.uint8)
    for i in range(8):
        res = cv2.bitwise_or(res, C[i, :, :] << i)

    if not os.path.exists('./result'):
        os.mkdir('./result')
    # save img
    save_path = os.path.join('./result', encrypt_img_path)
    cv2.imwrite(save_path, res)

    # save params
    if not use_params:
        A11_0_int = int('1' + ''.join([str(i) for i in A11_0[::-1]]), 2)
        A22_0_int = int('1' + ''.join([str(i) for i in A22_0[::-1]]), 2)
        if not os.path.exists('./params'):
            os.mkdir('./params')
        np.savez(f'./params/{params_path}', x0=x0, u1=u1, y0=y0, u2=u2, A11_0=A11_0_int, A22_0=A22_0_int,
                 n_round=n_round)
    print("ENCRYPTION DONE!")
    return res, save_path


if __name__ == '__main__':
    res, _ = encrypt('../images/grey/blonde.jpg', use_params=True)
    print(res.shape)