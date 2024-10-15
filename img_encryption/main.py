import cv2
import numpy as np

# 读取RGB图像
image = cv2.imread('D:\\todo\img_encryption\images\\rgb\lena.png')  # 替换为你的图像路径

# 检查图像是否成功读取
if image is None:
    raise ValueError("图像读取失败，请检查图像路径。")

# 获取图像的尺寸
height, width, channels = image.shape

# 分离通道
B, G, R = cv2.split(image)

# 将每一列的B、G、R值组合成一个三通道的列像素矩阵（height, width, 3）
columns = np.concatenate(np.array([B, G, R]), axis=1)  # 将每列B、G、R值合并
print(columns.shape)
# 随机生成下标序列，用于打乱列
random_indices = np.random.permutation(width*3)  # 生成打乱的列下标序列

# 按照打乱的下标重新排列列像素
shuffled_columns = np.empty_like(columns)
shuffled_columns[:, random_indices] = columns
print(shuffled_columns.shape)

# 从打乱的列重新组建RGB图像
shuffled_image = cv2.merge([shuffled_columns[:, :width], shuffled_columns[:, width:width*2], shuffled_columns[:, width*2:]])
print(shuffled_image.shape)

# 显示原图和打乱后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Shuffled Image', shuffled_image)



# 保存打乱后的图像
# cv2.imwrite('shuffled_image.jpg', shuffled_image)

shuffled_columns = shuffled_columns[:, random_indices]

# 从打乱的列重新组建RGB图像
shuffled_image = cv2.merge([shuffled_columns[:, :width], shuffled_columns[:, width:width*2], shuffled_columns[:, width*2:]])
print(type(shuffled_image))
cv2.imshow('reveal Image', shuffled_image)

# 等待按键操作
cv2.waitKey(0)
cv2.destroyAllWindows()
