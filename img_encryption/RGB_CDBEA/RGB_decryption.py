# -*-coding:utf-8 -*-

"""
# File       : RGB_decryption
# Time       ：2024/10/11 17:02
# Author     ：chenyu
# version    ：python 3.8
# Description：
"""
import cv2
import numpy as np
from decrypt import decrypt
from chaos.PWLCW import PWLCM


def gen_keys(keys, len):
    x0, h0, rounds, discard = keys['x1'], keys['h1'], keys['rounds'], keys['discard']
    seq = PWLCM(x0, h0, discard + len)[discard:]
    return seq


def scramble_reverse(img, keys, axis=0):
    h, w, c = img.shape
    if axis == 0:  # row
        length = h * 3
        keys_seq = gen_keys(keys, length)
        l_idx = np.argsort(keys_seq)
        # 分离通道
        B, G, R = cv2.split(img)
        rows = np.concatenate(np.array([B, G, R]), axis=0)  # 将每列B、G、R值合并
        reveal_rows = rows[l_idx, :]
        shuffled_image = cv2.merge(
            [reveal_rows[:h, :], reveal_rows[h:h * 2, :], reveal_rows[h * 2:h * 3, :]])
    else:  # column
        length = w * 3
        keys_seq = gen_keys(keys, length)
        l_idx = np.argsort(keys_seq)
        # 分离通道
        B, G, R = cv2.split(img)
        columns = np.concatenate(np.array([B, G, R]), axis=1)  # 将每列B、G、R值合并
        reveal_columns = columns[:, l_idx]
        shuffled_image = cv2.merge(
            [reveal_columns[:, :w], reveal_columns[:, w:w * 2], reveal_columns[:, w * 2:]])
    return shuffled_image


def decrypt_RGB(encrypt_img_path, raw_img_path, params_path='params.npz'):
    # load key
    npz_file = np.load(f"./params/{params_path}")
    x0, h0, rounds0, discard0, x1, h1, rounds1, discard1 = \
        npz_file['x0'], \
        npz_file['h0'], \
        npz_file['rounds0'], \
        npz_file['discard0'], \
        npz_file['x1'], \
        npz_file['h1'], \
        npz_file['rounds1'], \
        npz_file['discard1']
    keys0 = {
        'x0': x0,
        'h0': h0,
        'rounds': rounds0,
        'discard': discard0
    }
    keys1 = {
        'x1': x1,
        'h1': h1,
        'rounds': rounds0,
        'discard': discard0
    }
    # get encrypted img
    filename = raw_img_path.split('/')[-1].split('.')
    filename, ext = filename[0], filename[-1]
    decrypt_img_path = f"{filename}_decrypt.{ext}"

    img = cv2.imread(encrypt_img_path)
    H, W, C = img.shape  # M: height, N: width
    print(f"DECRYPTION: img resolution is {H} * {W}")
    B, G, R = cv2.split(img)
    # print(keys0)
    B = decrypt(B, keys0)
    G = decrypt(G, keys0)
    R = decrypt(R, keys0)
    res = cv2.merge([B, G, R])
    # RGB row and column scrambling reveal
    img = scramble_reverse(res, keys1, 0)
    res = scramble_reverse(img, keys1, 1)

    # cv2.imshow('reveal Image', res)
    # # 等待按键操作
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite('./result/' + decrypt_img_path, res)
    print('DECRYPTION DONE!')
    # 验证解密得到的图像为原图像
    if raw_img_path:
        raw_img = cv2.imread(raw_img_path)
        if not np.sum(np.abs(res - raw_img)):
            print("DECRYPTION SUCCESS!")
        else:
            print("DECRYPTION FAILURE!")

    return res


if __name__ == '__main__':
    decrypt_RGB('./result/lena_encrypt.png', '../images/rgb/lena.png')
