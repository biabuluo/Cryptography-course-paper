# -*-coding:utf-8 -*-

"""
# File       : main
# Time       ：2024/10/11 19:05
# Author     ：chenyu
# version    ：python 3.8
# Description：
"""
import matplotlib.pyplot as plt

from RGB_encryption import encrypt_RGB
from RGB_decryption import decrypt_RGB
import time

start = time.time()
img_path = "../images/rgb/lena.png"  # 图像路径
res, path = encrypt_RGB(img_path)
res_decrypted = decrypt_RGB(path, img_path)

end = time.time()
print(end-start)
origin = plt.imread(img_path)
plt.figure(figsize=(9, 3), dpi=300)
plt.subplot(131)
plt.title('origin')
plt.axis('off')
plt.imshow(origin)
plt.subplot(132)
plt.title('encrypted')
plt.axis('off')
plt.imshow(res)
plt.subplot(133)
plt.title('decrypted')
plt.axis('off')
res_decrypted = res_decrypted[:, :, ::-1]  # BGR -> RGB
plt.imshow(res_decrypted)

plt.savefig('./result/ex.jpg')