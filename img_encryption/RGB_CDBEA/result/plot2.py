# -*-coding:utf-8 -*-

"""
# File       : plot2
# Time       ：2024/10/15 20:18
# Author     ：chenyu
# version    ：python 3.8
# Description：
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import randint
from skimage import io

# 读取图片并提取RGB通道
image_path = './lena_encrypt.png'  # 替换为你的图片路径
img = io.imread(image_path)
height, width, _ = img.shape

# 分离RGB通道
R = img[:, :, 0]
G = img[:, :, 1]
B = img[:, :, 2]

# 随机采样2000对像素
n_samples = 2000


def calculate_pixel_pairs(channel, direction):
    pixel_pairs = []
    for _ in range(n_samples):
        x = randint(0, height - 2)
        y = randint(0, width - 2)

        if direction == 'horizontal':
            pixel1 = channel[x, y]
            pixel2 = channel[x, y + 1]
        elif direction == 'vertical':
            pixel1 = channel[x, y]
            pixel2 = channel[x + 1, y]
        elif direction == 'diagonal':
            pixel1 = channel[x, y]
            pixel2 = channel[x + 1, y + 1]

        pixel_pairs.append((pixel1, pixel2))

    return pixel_pairs


# 计算每个通道三个方向的像素对
directions = ['horizontal', 'vertical', 'diagonal']
pixel_pairs_R = [calculate_pixel_pairs(R, d) for d in directions]
pixel_pairs_G = [calculate_pixel_pairs(G, d) for d in directions]
pixel_pairs_B = [calculate_pixel_pairs(B, d) for d in directions]

# 创建子图
fig = plt.figure(figsize=(15, 5))

# 绘制红色通道的散点图
ax1 = fig.add_subplot(131, projection='3d')
for j, pairs in enumerate(pixel_pairs_R):
    x = np.array([p[0] for p in pairs])  # 第一个像素值
    y = np.array([p[1] for p in pairs])  # 第二个像素值
    z = np.full_like(x, j)  # 用 j 表示不同的方向 (0: horizontal, 1: vertical, 2: diagonal)

    ax1.scatter(x, y, z, color='r', alpha=0.1)
ax1.set_xlabel('Pixel 1 Value')
ax1.set_ylabel('Pixel 2 Value')
ax1.set_zlabel('Direction')
ax1.set_zticks([0, 1, 2])
# ax1.set_zticklabels(['Horizontal', 'Vertical', 'Diagonal'])
ax1.set_title('R Channel Correlation')

# 绘制绿色通道的散点图
ax2 = fig.add_subplot(132, projection='3d')
for j, pairs in enumerate(pixel_pairs_G):
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    z = np.full_like(x, j)

    ax2.scatter(x, y, z, color='g', alpha=0.1)
ax2.set_xlabel('Pixel 1 Value')
ax2.set_ylabel('Pixel 2 Value')
ax2.set_zlabel('Direction')
ax2.set_zticks([0, 1, 2])
# ax2.set_zticklabels(['Horizontal', 'Vertical', 'Diagonal'])
ax2.set_title('G Channel Correlation')

# 绘制蓝色通道的散点图
ax3 = fig.add_subplot(133, projection='3d')
for j, pairs in enumerate(pixel_pairs_B):
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    z = np.full_like(x, j)

    ax3.scatter(x, y, z, color='b', alpha=0.1)
ax3.set_xlabel('Pixel 1 Value')
ax3.set_ylabel('Pixel 2 Value')
ax3.set_zlabel('Direction')
ax3.set_zticks([0, 1, 2])
ax3.set_zticklabels(['Horizontal', 'Vertical', 'Diagonal'])
ax3.set_title('B Channel Correlation')

# 调整布局并显示
plt.tight_layout()
plt.show()
