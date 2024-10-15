# -*-coding:utf-8 -*-

"""
# File       : experiment
# Time       ：2024/10/15 18:09
# Author     ：chenyu
# version    ：python 3.8
# Description：
"""
from random import uniform
from BCIEA.utils import *
import numpy as np
from math import floor
from copy import deepcopy

def encrypt(
        A1, A2, M, N,
        n_round=1,  # epochs
):
    # initialization
    # x0 = uniform(1e-16, 1 - 1e-16)
    # u1 = uniform(1e-16, 0.5 - 1e-16)
    # y0 = uniform(1e-16, 1 - 1e-16)
    # u2 = uniform(1e-16, 0.5 - 1e-16)
    x0 = 0.6952020325739682
    u1 = 0.06975026107356343
    y0 = 0.6952020325739682
    u2 = 0.06975026107356343

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

    return A1, A2