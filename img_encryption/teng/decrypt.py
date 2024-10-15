# -*-coding:utf-8 -*-

"""
# File       : decrypt
# Time       ：2024/10/14 18:53
# Author     ：chenyu
# version    ：python 3.8
# Description：teng 解密
"""
from math import floor
from copy import deepcopy

from teng.utils import *
from teng.chaos import *

def decrypt(encrypt_img_path, raw_img_path=None, params_path='params.npz'):
    # load key
    npz_file = np.load(f"./params/{params_path}")
    alpha0, sigma0, y0, alpha1, init_l = \
        npz_file['alpha0'], \
        npz_file['sigma0'], \
        npz_file['y0'], \
        npz_file['alpha1'], \
        npz_file['init_l']

    # get encrypted img
    filename = raw_img_path.split('/')[-1].split('.')
    filename, ext = filename[0], filename[-1]
    decrypt_img_path = f"{filename}_decrypt.{ext}"

    img = cv2.imread(encrypt_img_path, cv2.IMREAD_GRAYSCALE)
    M, N = img.shape                                    # M: height, N: width
    length = M * N
    print(f"DECRYPTION: img resolution is {N} * {M}")

    D = 1000                                           # discard num

    bitplanes = img_bit_decomposition(img)  # Image bit plane decomposition
    bitplanes = bitplanes[::-1, :, :]  # reverse
    # high bits
    C8 = bitplanes[0, :, :].ravel()
    C7 = bitplanes[1, :, :].ravel()
    C6 = bitplanes[2, :, :].ravel()
    C5 = bitplanes[3, :, :].ravel()
    # low
    C4 = bitplanes[4, :, :].ravel()
    C3 = bitplanes[5, :, :].ravel()
    C2 = bitplanes[6, :, :].ravel()
    C1 = bitplanes[7, :, :].ravel()

    chaos_seq = np.array(CML_map(alpha0, sigma0, init_l, length, D))
    TN = np.array([np.argsort(x) for x in chaos_seq])
    Y = np.array(logistic_map(alpha1, y0, 4 * length + D)[D:])
    TM = np.argsort(Y)
    C = np.concatenate([C1, C2, C3, C4])
    C = C[TM]
    B1 = C[:length]
    B2 = C[length:2 * length]
    B3 = C[2 * length:3 * length]
    B4 = C[3 * length:]

    Z = np.round(Y[:length] * 1e8).astype(np.uint8) % 256
    Z = Z.reshape(M, N)
    # print(Z.shape)
    R = img_bit_decomposition(Z)

    B5_temp = (C5 - B1 - R[4].reshape(length)) % 2
    B6_temp = (C6 - B2 - R[5].reshape(length)) % 2
    B7_temp = (C7 - B3 - R[6].reshape(length)) % 2
    B8_temp = (C8 - B4 - R[7].reshape(length)) % 2

    B5 = B5_temp[TN[0]]
    B6 = B6_temp[TN[1]]
    B7 = B7_temp[TN[2]]
    B8 = B8_temp[TN[3]]

    # composition the encrypted img
    B = np.stack([B1.reshape(M, N),
                  B2.reshape(M, N),
                  B3.reshape(M, N),
                  B4.reshape(M, N),
                  B5.reshape(M, N),
                  B6.reshape(M, N),
                  B7.reshape(M, N),
                  B8.reshape(M, N)], axis=0)
    res = np.zeros((M, N), dtype=np.uint8)
    for i in range(8):
        res = cv2.bitwise_or(res, B[i, :, :] << i)

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
    encrypt_img_path = "./result/med_encrypt.png"  # 要解密的图像路径
    raw_img_path = "../images/grey/med.png"  # 原始图像路径
    decrypt(encrypt_img_path, raw_img_path)
