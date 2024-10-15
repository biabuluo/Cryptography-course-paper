# -*-coding:utf-8 -*-

"""
# File       : decrypt
# Time       ：2024/10/6 20:19
# Author     ：chenyu
# version    ：python 3.8
# Description：
"""
from math import floor
import cv2
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from BCIEA.utils import PWLCM, img_bit_decomposition


def decrypt(encrypt_img_path, raw_img_path=None, params_path='params.npz'):
    # load key
    npz_file = np.load(f"./params/{params_path}")
    x0, u1, y0, u2, A11_0, A22_0, n_round = \
        npz_file['x0'], \
        npz_file['u1'], \
        npz_file['y0'], \
        npz_file['u2'], \
        npz_file['A11_0'], \
        npz_file['A22_0'], \
        npz_file['n_round']
    A11_0 = [int(i) for i in list(bin(A11_0)[2:])[1:]]
    A22_0 = [int(i) for i in list(bin(A22_0)[2:])[1:]]

    # get encrypted img
    filename = raw_img_path.split('/')[-1].split('.')
    filename, ext = filename[0], filename[-1]
    decrypt_img_path = f"{filename}_decrypt.{ext}"

    img = cv2.imread(encrypt_img_path, cv2.IMREAD_GRAYSCALE)
    M, N = img.shape                                    # M: height, N: width
    print(f"DECRYPTION: img resolution is {N} * {M}")

    # Piecewise Logistic Chaotic Map
    N0 = 1000                                           # discard num
    PWLCM_MAP_X = PWLCM(x0, u1, N0 + M*N)[N0:]
    X1 = [floor(i * 1e14) % 256 for i in PWLCM_MAP_X]   # key1
    X1_reshape = np.mat(X1, dtype=np.uint8).reshape(M, N)
    X1_bitplanes = img_bit_decomposition(X1_reshape)    # key1 discomposition
    b1 = X1_bitplanes[::2, :, :].ravel()                # key1 even bits
    b2 = X1_bitplanes[1::2, :, :].ravel()               # key1 odd bits

    bitplanes = img_bit_decomposition(img)              # img discomposition
    C1 = bitplanes[:4, :, :].ravel()
    C2 = bitplanes[4:, :, :].ravel()

    # start decryption
    L = M * N * 4
    for k in range(n_round):
        # anti-confusion
        sum = np.sum(C1) + np.sum(C2)       # C1 和 C2 的和
        s0 = (y0 + sum / L) % 1             # key2的初始值s0
        S = PWLCM(s0, u2, N0 + 2 * L)[N0:]
        S1, S2 = S[:L], S[L:]
        Y = [floor(s1 * 1e14) % L for s1 in S1]
        Z = [floor(s2 * 1e14) % L for s2 in S2]

        for j in range(L-1, -1, -1):
            C2[j], C1[Z[j]] = C1[Z[j]], C2[j]
        for i in range(L-1, -1, -1):
            C1[i], C2[Y[i]] = C2[Y[i]], C1[i]

        B1, B2 = C1, C2

        # anti-diffusion
        sum2 = np.sum(B1)  # B1 的和
        A11 = np.zeros(L, dtype=np.uint8)
        A22 = np.zeros(L, dtype=np.uint8)
        A22[0] = A22_0[k]
        for i in range(1, L):
            A22[i] = A22[i - 1] ^ B1[i] ^ B2[i] ^ b2[i]
        A2 = np.roll(A22, -1*sum2)  # A2 循环左移 sum2 位
        sum1 = np.sum(A2)  # A2 的和
        A11[0] = A11_0[k]
        for i in range(1, L):
            A11[i] = A11[i - 1] ^ A2[i] ^ B1[i] ^ b1[i]
        A1 = np.roll(A11, -1*sum1)  # A1 循环左移 sum1 位
        C1 = deepcopy(A1)
        C2 = deepcopy(A2)

    # A1，A2两个序列合并成原始图像
    A = np.append(A1, A2)[::-1].reshape(8, M, N)[:, ::-1, ::-1]
    res = np.zeros((M, N), dtype=np.uint8)
    for i in range(8):
        res = cv2.bitwise_or(res, A[i, :, :] << i)

    cv2.imwrite('./result/'+decrypt_img_path, res)
    print('DECRYPTION DONE!')
    # 验证解密得到的图像为原图像
    if raw_img_path:
        raw_img = cv2.imread(raw_img_path, cv2.IMREAD_GRAYSCALE)
        if not np.sum(np.abs(res - raw_img)):
            print("DECRYPTION SUCCESS!")
        else:
            print("DECRYPTION FAILURE!")

    return res


if __name__ == "__main__":
    encrypt_img_path = "./result/blonde_encrypt.jpg"  # 要解密的图像路径
    raw_img_path = "../images/grey/blonde.jpg"  # 原始图像路径
    decrypt(encrypt_img_path, raw_img_path)
