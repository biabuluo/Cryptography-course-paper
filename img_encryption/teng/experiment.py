# -*-coding:utf-8 -*-

"""
# File       : experiment
# Time       ：2024/10/14 15:36
# Author     ：chenyu
# version    ：python 3.8
# Description：
"""
import numpy as np
from teng.utils import *
from teng.chaos import *

def get_distence(a, b):
    return np.sum(np.abs(a - b))


# H, W = 64, 64
# length = H * W

def encrypt(a1, a2, H, W):
    length = H * W
    b1 = a1[:length]
    b2 = a1[length:2*length]
    b3 = a1[2*length:3*length]
    b4 = a1[3*length:]

    b5 = a2[:length]
    b6 = a2[length:2*length]
    b7 = a2[2*length:3*length]
    b8 = a2[3*length:]

    img = np.stack([b1, b2, b3, b4, b5, b6, b7, b8], axis=0)

    # step 3
    init_l = []
    for i in range(4):
        init_l.append(get_init(img[i], i))

    alpha = 3.7
    sigma = 0.5
    M = 100
    chaos_seq = np.array(CML_map(alpha, sigma, init_l, length, M))
    # print(np.array(chaos_seq).shape)

    TN = np.array([np.argsort(x) for x in chaos_seq])
    # print(TN.shape)

    b5_temp = np.empty_like(b5)
    b5_temp[TN[0]] = b5
    b6_temp = np.empty_like(b6)
    b6_temp[TN[1]] = b6
    b7_temp = np.empty_like(b7)
    b7_temp[TN[2]] = b7
    b8_temp = np.empty_like(b8)
    b8_temp[TN[3]] = b8

    # diffusion
    alpha1 = 3.8
    y0 = 0.6
    m1 = 500
    Y = np.array(logistic_map(alpha1, y0, 4 * length + m1)[m1:])
    Z = np.round(Y[:length] * 1e8).astype(np.uint8) % 256
    Z = Z.reshape(H, W)
    # print(Z.shape)
    R = img_bit_decomposition(Z)
    # print(R.shape)

    C5 = (b5_temp + b1 + R[4].reshape(length)) % 2
    C6 = (b6_temp + b2 + R[5].reshape(length)) % 2
    C7 = (b7_temp + b3 + R[6].reshape(length)) % 2
    C8 = (b8_temp + b4 + R[7].reshape(length)) % 2

    TM = np.argsort(Y)
    # print(TM.shape)
    C = np.concatenate([b1, b2, b3, b4])
    # print(C.shape)
    C_temp = np.empty_like(C)
    C_temp[TM] = C

    C1 = C_temp[:length]
    C2 = C_temp[length:2 * length]
    C3 = C_temp[2 * length:3 * length]
    C4 = C_temp[3 * length:]
    return np.concatenate([C1, C2, C3, C4], axis=0), np.concatenate([C5, C6, C7, C8], axis=0)


if __name__ == '__main__':
    H, W = 64, 64
    length = 4 * H * W
    a1 = np.random.randint(2, size=length)
    a2 = np.random.randint(2, size=length)
    a, b = encrypt(a1, a2, H, W)
    print(a.shape)
    print(b.shape)
# b1 = np.random.randint(2, size=length)
# b2 = np.random.randint(2, size=length)
# b3 = np.random.randint(2, size=length)
# b4 = np.random.randint(2, size=length)

# b5 = np.random.randint(2, size=length)
# b6 = np.random.randint(2, size=length)
# b7 = np.random.randint(2, size=length)
# b8 = np.random.randint(2, size=length)

# img = np.stack([b1, b2, b3, b4, b5, b6, b7, b8], axis=0)
#
# # step 3
# init_l = []
# for i in range(4):
#     init_l.append(get_init(img[i], i))
#
# alpha = 3.7
# sigma = 0.5
# M = 100
# chaos_seq = np.array(CML_map(alpha, sigma, init_l, length, M))
# # print(np.array(chaos_seq).shape)
#
# TN = np.array([np.argsort(x) for x in chaos_seq])
# # print(TN.shape)
#
# b5_temp = np.empty_like(b5)
# b5_temp[TN[0]] = b5
# b6_temp = np.empty_like(b6)
# b6_temp[TN[1]] = b6
# b7_temp = np.empty_like(b7)
# b7_temp[TN[2]] = b7
# b8_temp = np.empty_like(b8)
# b8_temp[TN[3]] = b8
#
# # diffusion
# alpha1 = 3.8
# y0 = 0.6
# m1 = 500
# Y = np.array(logistic_map(alpha1, y0, 4*length+m1)[m1:])
# Z = np.round(Y[:length]*1e8).astype(np.uint8) % 256
# Z = Z.reshape(H, W)
# # print(Z.shape)
# R = img_bit_decomposition(Z)
# # print(R.shape)
#
# C5 = (b5_temp + b1 + R[4].reshape(length)) % 2
# C6 = (b6_temp + b2 + R[5].reshape(length)) % 2
# C7 = (b7_temp + b3 + R[6].reshape(length)) % 2
# C8 = (b8_temp + b4 + R[7].reshape(length)) % 2
#
# TM = np.argsort(Y)
# # print(TM.shape)
# C = np.concatenate([b1, b2, b3, b4])
# # print(C.shape)
# C_temp = np.empty_like(C)
# C_temp[TM] = C
#
# C1 = C_temp[:length]
# C2 = C_temp[length:2*length]
# C3 = C_temp[2*length:3*length]
# C4 = C_temp[3*length:]
#
# total = get_distence(b1, C1)+get_distence(b2, C2)+get_distence(b3, C3)+ \
#         get_distence(b4, C4) + get_distence(b5, C5) + get_distence(b6, C6) + \
#         get_distence(b7, C7) + get_distence(b8, C8)
#
# print(total/8)

# decrypted
# C = np.concatenate([C1, C2, C3, C4])
# C = C[TM]
# B1 = C[:length]
# B2 = C[length:2*length]
# B3 = C[2*length:3*length]
# B4 = C[3*length:]
# # total = get_distence(b1, B1)+get_distence(b2, B2)+get_distence(b3, B3)+ \
# #         get_distence(b4, B4)
# #
# # print(total)
# B5_temp = (C5 - B1 - R[4].reshape(length)) % 2
# B6_temp = (C6 - B2 - R[5].reshape(length)) % 2
# B7_temp = (C7 - B3 - R[6].reshape(length)) % 2
# B8_temp = (C8 - B4 - R[7].reshape(length)) % 2
# # total = get_distence(B5_temp, b5_temp)+get_distence(B6_temp, b6_temp)+get_distence(B7_temp, b7_temp)+ \
# #         get_distence(B8_temp, b8_temp)
# #
# # print(total)
#
# B5 = B5_temp[TN[0]]
# B6 = B6_temp[TN[1]]
# B7 = B7_temp[TN[2]]
# B8 = B8_temp[TN[3]]
# total = get_distence(B5, b5)+get_distence(B6, b6)+get_distence(B7, b7)+ \
#         get_distence(B8, b8)
#
# print(total)