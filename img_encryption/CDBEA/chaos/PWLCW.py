# -*-coding:utf-8 -*-

"""
# File       : PWLCW
# Time       ：2024/10/7 0:52
# Author     ：chenyu
# version    ：python 3.8
# Description：
"""
# chaotic map
# x0:init value; h:control para; num:len of the seq
def PWLCM(x0, h, num):
    pwlcm = [0] * num
    pwlcm[0] = x0
    for i in range(1, num):
        if 0 < pwlcm[i - 1] < h:
            pwlcm[i] = pwlcm[i - 1] / h
        elif h <= pwlcm[i - 1] < 0.5:
            pwlcm[i] = (pwlcm[i - 1] - h) / (0.5 - h)
        elif 0.5 <= pwlcm[i - 1] < 1 - h:
            pwlcm[i] = (1 - pwlcm[i - 1] - h) / (0.5 - h)
        elif 1 - h <= pwlcm[i - 1] < 1:
            pwlcm[i] = (1 - pwlcm[i - 1]) / h
        else:
            raise ValueError("xi must be in [0, 1]")
    return pwlcm


if __name__ == '__main__':
    seq1 = PWLCM(0.2, 1e-16, 4*512*512)
    seq2 = PWLCM(0.2, 1e-16, 4 * 512 * 512)
