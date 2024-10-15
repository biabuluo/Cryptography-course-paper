# -*-coding:utf-8 -*-

"""
# File       : plot
# Time       ：2024/10/15 19:55
# Author     ：chenyu
# version    ：python 3.8
# Description：
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

# 加载图像
image_path = './lena_encrypt.png'  # 替换为你的图像路径
img = Image.open(image_path)
img = np.array(img)

# 提取RGB通道
r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

# 计算每个通道的直方图
hist_r, bins_r = np.histogram(r.flatten(), bins=256, range=[0, 256])
hist_g, bins_g = np.histogram(g.flatten(), bins=256, range=[0, 256])
hist_b, bins_b = np.histogram(b.flatten(), bins=256, range=[0, 256])

# 准备绘图
fig = plt.figure(figsize=(8, 4))

# 左侧显示RGB原图
ax1 = fig.add_subplot(121)
ax1.imshow(img)
ax1.set_title('Original RGB Image')
ax1.axis('off')  # 关闭坐标轴

# 右侧显示3D直方图
ax2 = fig.add_subplot(122, projection='3d')

# 准备3D坐标数据
bin_centers = np.arange(256)

# 绘制红色通道直方图
ax2.bar(bin_centers, hist_r, zs=0, zdir='y', color='r', alpha=0.7, label='Red Channel')

# 绘制绿色通道直方图
ax2.bar(bin_centers, hist_g, zs=1, zdir='y', color='g', alpha=0.7, label='Green Channel')

# 绘制蓝色通道直方图
ax2.bar(bin_centers, hist_b, zs=2, zdir='y', color='b', alpha=0.7, label='Blue Channel')

# 设置轴标签
ax2.set_xlabel('Pixel Value')
ax2.set_ylabel('Channel')
ax2.set_zlabel('Frequency')
ax2.set_yticks([0, 1, 2])
ax2.set_yticklabels(['Red', 'Green', 'Blue'])
# ax2.legend()
ax2.set_title('3D Histogram of RGB Channels')

plt.tight_layout()
plt.show()
