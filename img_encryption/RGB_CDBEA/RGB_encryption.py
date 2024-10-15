# -*-coding:utf-8 -*-

"""
# File       : RGB_excryption
# Time       ：2024/10/10 23:07
# Author     ：chenyu
# version    ：python 3.8
# Description：彩图扩展加密
"""
from math import floor
import cv2
import numpy as np
import os
from random import uniform

from encrypt import encrypt
from chaos.PWLCW import PWLCM

def gen_keys(keys, len):
    x0, h0, rounds, discard = keys['x1'], keys['h1'], keys['rounds'], keys['discard']
    seq = PWLCM(x0, h0, discard + len)[discard:]
    return seq

def scramble(img, keys, axis=0):
    h, w, c = img.shape
    if axis == 0:  # row
        length = h*3
        keys_seq = gen_keys(keys, length)
        l_idx = np.argsort(keys_seq)
        # 分离通道
        B, G, R = cv2.split(img)
        rows = np.concatenate(np.array([B, G, R]), axis=0)  # 将每列B、G、R值合并
        # 按照打乱的下标重新排列列像素
        shuffled_rows = np.empty_like(rows)
        shuffled_rows[l_idx, :] = rows
        shuffled_image = cv2.merge(
            [shuffled_rows[:h, :], shuffled_rows[h:h*2, :], shuffled_rows[h*2:h*3, :]])
    else:          # column
        length = w*3
        keys_seq = gen_keys(keys, length)
        l_idx = np.argsort(keys_seq)
        # 分离通道
        B, G, R = cv2.split(img)
        columns = np.concatenate(np.array([B, G, R]), axis=1)  # 将每列B、G、R值合并
        # 按照打乱的下标重新排列列像素
        shuffled_columns = np.empty_like(columns)
        shuffled_columns[:, l_idx] = columns
        shuffled_image = cv2.merge(
            [shuffled_columns[:, :w], shuffled_columns[:, w:w * 2], shuffled_columns[:, w * 2:]])
    return shuffled_image


def encrypt_RGB(
        img_path,
        x0=None, h0=None, rounds0=None, discard0=None,  # keys
        x1=None, h1=None, rounds1=None, discard1=None,  # RGB_keys
        params_path='params.npz',  # the path of saved params
        use_params=False,  # if to use used params
):
    # initialization
    if not use_params:
        x0 = uniform(1e-16, 1 - 1e-16) if not x0 else x0
        h0 = uniform(1e-16, 0.5 - 1e-16) if not h0 else h0
        rounds0 = 1 if not rounds0 else rounds0
        discard0 = 1000 if not discard0 else discard0
        x1 = uniform(1e-16, 1 - 1e-16) if not x1 else x1
        h1 = uniform(1e-16, 0.5 - 1e-16) if not h1 else h1
        rounds1 = 1 if not rounds1 else rounds1
        discard1 = 1000 if not discard1 else discard1
    else:
        use_params_path = f"./params/{params_path}"
        if os.path.exists(use_params_path):
            params = np.load(use_params_path)
            x0 = params['x0']
            h0 = params['h0']
            x1 = params['x1']
            h1 = params['h1']
            rounds0 = params['rounds0']
            discard0 = params['discard0']
        else:
            raise FileNotFoundError(f"can not find the params file: {use_params_path}")

    filename = img_path.split('/')[-1].split('.')
    filename, ext = filename[0], filename[-1]
    encrypt_img_path = f"{filename}_encrypt.png"
    # get img (grey img supported)
    img = cv2.imread(img_path)
    # 检查图像是否为RGB
    if img.shape[2] != 3:
        raise ValueError("图像必须是RGB格式")
    H, W, C = img.shape
    length = 4 * 1 * H * W
    print(f"ENCRYPTION: img resolution is {H} * {W}")

    # get params
    keys0 = {
        'x0': x0,
        'h0': h0,
        'rounds': rounds0,
        'discard': discard0
    }
    keys1 = {
        'x1': x1,
        'h1': h1,
        'rounds': rounds1,
        'discard': discard1
    }
    # RGB row and column scrambling
    img = scramble(img, keys1, 1)
    # cv2.imshow('0', img)
    # # 等待按键操作
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img = scramble(img, keys1, 0)
    # cv2.imshow('1', img)
    # # 等待按键操作
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    B, G, R = cv2.split(img)
    B = encrypt(B, keys0)
    G = encrypt(G, keys0)
    R = encrypt(R, keys0)
    res = cv2.merge([B, G, R])

    # cv2.imshow('encrypted Image', res)
    # # 等待按键操作
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if not os.path.exists('./result'):
        os.mkdir('./result')
    # save img
    save_path = os.path.join('./result', encrypt_img_path)
    cv2.imwrite(save_path, res)

    # save params
    if not use_params:
        if not os.path.exists('./params'):
            os.mkdir('./params')
        np.savez(f'./params/{params_path}', x0=x0, h0=h0, rounds0=rounds0, discard0=discard0,
                 x1=x1, h1=h1, rounds1=rounds1, discard1=discard1)
    print("ENCRYPTION DONE!")
    return res, save_path


if __name__ == '__main__':
    encrypt_RGB('../images/rgb/lena.png')

