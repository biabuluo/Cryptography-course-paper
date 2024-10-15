# -*-coding:utf-8 -*-

"""
# File       : chaos
# Time       ：2024/10/14 15:29
# Author     ：chenyu
# version    ：python 3.8
# Description：cml + logistic map
"""
import numpy as np


def logistic_map(alpha, x0, length):
    assert 3.5699456 < alpha < 4
    assert 0 < x0 < 1
    res = []
    for _ in range(length):
        if _ == 0:
            res.append(alpha * x0 * (1 - x0))
        else:
            res.append(alpha * res[-1] * (1 - res[-1]))
    return res


def iter_logistic_map(alpha, x):
    return alpha * x * (1 - x)


def CML_map(alpha, sigma, init_l, len_res, discard):
    assert 0 < sigma < 1
    length = len(init_l)
    res = []
    for i in range(length):
        res.append([])
    for i in range(len_res+discard):
        for j in range(length):
            if i == 0:
                a = iter_logistic_map(alpha, init_l[j])
                b = iter_logistic_map(alpha, init_l[j - 1])
                c = iter_logistic_map(alpha, init_l[(j + 1) % length])
                temp = (1 - sigma) * a + (sigma / 2) * (b + c)
                res[j].append(temp)
            else:
                a = iter_logistic_map(alpha, res[j][-1])
                b = iter_logistic_map(alpha, res[j - 1][-1])
                c = iter_logistic_map(alpha, res[(j + 1) % length][-1])
                temp = (1 - sigma) * a + (sigma / 2) * (b + c)
                res[j].append(temp)
    return [i[discard:] for i in res]


def get_init(l, idx):
    return 1 / ((sum(l) + 1) * (idx + 1))


if __name__ == '__main__':
    alpha = 3.7
    sigma = 0.5
    len_res = 10
    init_l = [0.1, 0.2, 0.3, 0.4]
    # res = CML_map(alpha, sigma, init_l, len_res, 100)
    res = logistic_map(alpha, 0.1, 100)
    print(np.array(res).shape)

