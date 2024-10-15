# -*-coding:utf-8 -*-

"""
# File       : encrypt
# Time       ：2024/10/14 17:26
# Author     ：chenyu
# version    ：python 3.8
# Description：teng 加密
"""
from random import uniform
import os
from teng.utils import *
from teng.chaos import *

def encrypt(
        img_path,
        alpha0=None, sigma0=None,  # keys
        alpha1=None, y0=None,  # keys
        params_path='params.npz',  # the path of saved params
        use_params=False,  # if to use used params
):
    D = 1000  # discard
    # initialization
    if not use_params:
        alpha0 = uniform(1e-16+3.5699456, 4 - 1e-16) if not alpha0 else alpha0
        sigma0 = uniform(1e-16, 1 - 1e-16) if not sigma0 else sigma0
        y0 = uniform(1e-16, 1 - 1e-16) if not y0 else y0
        alpha1 = uniform(1e-16+3.5699456, 4 - 1e-16) if not alpha1 else alpha1
    else:
        use_params_path = f"./params/{params_path}"
        if os.path.exists(use_params_path):
            params = np.load(use_params_path)
            alpha0 = params['alpha0']
            sigma0 = params['sigma0']
            y0 = params['y0']
            alpha1 = params['alpha1']
        else:
            raise FileNotFoundError(f"can not find the params file: {use_params_path}")

    filename = img_path.split('/')[-1].split('.')
    filename, ext = filename[0], filename[-1]
    encrypt_img_path = f"{filename}_encrypt.png"
    # get img (grey img supported)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    M, N = img.shape  # M: height, N: width
    length = M * N
    print(f"ENCRYPTION: img resolution is {N} * {M}")

    bitplanes = img_bit_decomposition(img)  # Image bit plane decomposition
    bitplanes = bitplanes[::-1, :, :]  # reverse
    # high bits
    b8 = bitplanes[0, :, :].ravel()
    b7 = bitplanes[1, :, :].ravel()
    b6 = bitplanes[2, :, :].ravel()
    b5 = bitplanes[3, :, :].ravel()
    # low
    b4 = bitplanes[4, :, :].ravel()
    b3 = bitplanes[5, :, :].ravel()
    b2 = bitplanes[6, :, :].ravel()
    b1 = bitplanes[7, :, :].ravel()

    img = np.stack([b1, b2, b3, b4, b5, b6, b7, b8], axis=0)

    # step 3
    init_l = []
    for i in range(4):
        init_l.append(get_init(img[i], i))

    chaos_seq = np.array(CML_map(alpha0, sigma0, init_l, length, D))
    TN = np.array([np.argsort(x) for x in chaos_seq])

    b5_temp = np.empty_like(b5)
    b5_temp[TN[0]] = b5
    b6_temp = np.empty_like(b6)
    b6_temp[TN[1]] = b6
    b7_temp = np.empty_like(b7)
    b7_temp[TN[2]] = b7
    b8_temp = np.empty_like(b8)
    b8_temp[TN[3]] = b8

    Y = np.array(logistic_map(alpha1, y0, 4 * length + D)[D:])
    Z = np.round(Y[:length] * 1e8).astype(np.uint8) % 256
    Z = Z.reshape(M, N)
    R = img_bit_decomposition(Z)

    C5 = (b5_temp + b1 + R[4].reshape(length)) % 2
    C6 = (b6_temp + b2 + R[5].reshape(length)) % 2
    C7 = (b7_temp + b3 + R[6].reshape(length)) % 2
    C8 = (b8_temp + b4 + R[7].reshape(length)) % 2

    TM = np.argsort(Y)
    C = np.concatenate([b1, b2, b3, b4])
    C_temp = np.empty_like(C)
    C_temp[TM] = C

    C1 = C_temp[:length]
    C2 = C_temp[length:2 * length]
    C3 = C_temp[2 * length:3 * length]
    C4 = C_temp[3 * length:]

    # composition the encrypted img
    C = np.stack([C1.reshape(M, N),
                  C2.reshape(M, N),
                  C3.reshape(M, N),
                  C4.reshape(M, N),
                  C5.reshape(M, N),
                  C6.reshape(M, N),
                  C7.reshape(M, N),
                  C8.reshape(M, N)], axis=0)
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
        if not os.path.exists('./params'):
            os.mkdir('./params')
        np.savez(f'./params/{params_path}',
                 alpha0=alpha0, sigma0=sigma0, alpha1=alpha1, y0=y0, init_l=init_l)
    print("ENCRYPTION DONE!")
    return res, save_path


if __name__ == '__main__':
    res, _ = encrypt('../images/grey/med.png', use_params=False)
    print(res.shape)