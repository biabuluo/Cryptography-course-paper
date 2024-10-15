# -*-coding:utf-8 -*-

"""
# File       : plot
# Time       ：2024/10/15 19:25
# Author     ：chenyu
# version    ：python 3.8
# Description：
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据
np.random.seed(42)
xs = np.random.rand(50)
ys = np.random.rand(50)
zs = np.random.rand(50)

colors = np.random.rand(50)
# sizes = 100 * np.random.rand(50)
sizes=np.array(50*[30])

# 绘制三维散点图
fig = plt.figure()
plt.rcParams['font.sans-serif'] = ['FangSong']
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs, s=sizes, c=colors, marker='o', alpha=0.7, cmap='viridis')
ax.set_title('用户特征画像')
ax.set_xlabel('属性1')
ax.set_ylabel('属性2')
ax.set_zlabel('属性3')
plt.show()


# img1 = cv2.imread("./blonde.jpg", 0)
# img2 = cv2.imread("./darkhair.jpg", 0)
# img3 = cv2.imread("./livingroom.jpg", 0)
# img4 = cv2.imread("./mandril.jpg", 0)
#
# ##显示图片
# plt.figure(figsize=(9, 3))
# plt.subplot(141)
# plt.imshow(img1, cmap="gray")
# plt.title('blonde')
# plt.subplot(142)
# plt.imshow(img2, cmap="gray")
# plt.title('darkhair')
# plt.subplot(143)
# plt.imshow(img3, cmap="gray")
# plt.title('livingroom')
# plt.subplot(144)
# plt.imshow(img4, cmap="gray")
# plt.title('mandril')
# plt.tight_layout()
# plt.show()

