# -*-coding:utf-8 -*-

"""
# File       : operator
# Time       ：2024/10/7 19:20
# Author     ：chenyu
# version    ：python 3.8
# Description：DNA算子
"""
import numpy as np
import copy as deepcopy

# 编码规则 A,C,G,T:0,1,2,3
Rules = [
    [0, 1, 2, 3],
    [0, 2, 1, 3],
    [3, 1, 2, 0],
    [3, 2, 1, 0],
    [1, 0, 3, 2],
    [1, 3, 0, 2],
    [2, 0, 3, 1],
    [2, 3, 0, 1]
]

# operator:0-6
# addition subtraction multiplication exclusive-or/nor
# 左移 右移
Operator = np.array([
    [
        [1, 0, 3, 2],
        [0, 1, 2, 3],
        [3, 2, 1, 0],
        [2, 3, 0, 1]
    ],
    [
        [1, 2, 3, 0],
        [0, 1, 2, 3],
        [3, 0, 1, 2],
        [2, 3, 0, 1]
    ],
    [
        [3, 2, 0, 1],
        [2, 3, 1, 0],
        [1, 0, 3, 2],
        [0, 1, 2, 3]
    ],
    [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0]
    ],
    [
        [3, 2, 1, 0],
        [2, 3, 0, 1],
        [1, 0, 3, 2],
        [0, 1, 2, 3]
    ],
    [
        [0, 1, 2, 3],
        [1, 2, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 1, 2]
    ],
    [
        [0, 3, 2, 1],
        [1, 0, 3, 2],
        [2, 1, 0, 3],
        [3, 2, 1, 0]
    ]
])

# 单碱基突变
def sbn(arr1, arr2, swap_mask):
    # 确保输入数组的长度相同
    if len(arr1) != len(arr2) or len(arr1) != len(swap_mask):
        raise ValueError("所有数组必须具有相同的长度")
    # 将 01 数组转换为布尔数组
    swap_mask = swap_mask.astype(bool)
    # 创建输出数组
    new_arr1 = np.copy(arr1)
    new_arr2 = np.copy(arr2)

    # 根据布尔掩码进行互换
    new_arr1[swap_mask], new_arr2[swap_mask] = arr2[swap_mask], arr1[swap_mask]

    return new_arr1, new_arr2

# permutation
def permutation(l1, l2, l_idx):
    # 绑定数组并根据下标数组进行置换
    combined = np.stack((l1, l2), axis=0)
    new_array = np.empty_like(combined)
    new_array[:, l_idx] = combined
    a, b = new_array
    return a, b

# permutation_reverse
def permutation_reverse(l1, l2, l_idx):
    # 绑定数组并根据下标数组进行置换
    combined = np.stack((l1, l2), axis=0)
    new_array = np.empty_like(combined)
    # 根据下标数组重排
    new_array = combined[:, l_idx]
    a, b = new_array
    return a, b


def DNA_encoding(l1, l2, r):
    x = l1 * 2 + l2
    assert max(x) < 4
    rule = np.array(Rules[r])
    result = rule[x]
    return result


def DNA_decoding(x, r):
    assert max(x) < 4
    rule = Rules[r]
    x = np.array([rule.index(i) for i in x], dtype=np.uint8)
    high_bits = (x//2)%2
    low_bits = x%2
    return high_bits, low_bits


# DNA 运算
def DNA_operation(l1, l2, l_op):
    # 获取每个操作的结果
    l_new = np.array([Operator[op][l1[i]][l2[i]] for i, op in enumerate(l_op)])
    return l_new
