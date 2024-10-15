# -*-coding:utf-8 -*-

"""
# File       : main
# Time       ：2024/10/14 19:10
# Author     ：chenyu
# version    ：python 3.8
# Description：
"""
import matplotlib.pyplot as plt

from encrypt import encrypt
from decrypt import decrypt
import time

start = time.time()
img_path = "../images/grey/med.png"  # 图像路径
params = {
    'alpha0': None,
    'sigma0': None,
    'y0': None,
    'alpha1': None
}
res, path = encrypt(img_path, **params)
res_decrpted = decrypt(path, img_path)

end = time.time()
print(end-start)
origin = plt.imread(img_path)
plt.figure(figsize=(9, 3), dpi=300)
plt.subplot(131)
plt.title('origin')
plt.axis('off')
plt.imshow(origin, cmap='gray')
plt.subplot(132)
plt.title('encrypted')
plt.axis('off')
plt.imshow(res, cmap='gray')
plt.subplot(133)
plt.title('decrypted')
plt.axis('off')
plt.imshow(res_decrpted, cmap='gray')

plt.savefig('./result/ex.jpg')
