# -*-coding:utf-8 -*-

"""
# File       : plot
# Time       ：2024/10/15 18:53
# Author     ：chenyu
# version    ：python 3.8
# Description：
"""
import csv
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('./size_iter.csv')
x = data['iter']
xu = data['xu']
proposed = data['proposed']
teng = data['teng']

plt.figure(figsize=(5, 3))
plt.plot(x, xu, label='Xu et al.', color='#5BB5AC')
plt.plot(x, proposed, label='proposed', color='#DE526C')
plt.plot(x, teng, label='Teng et al.', color='#D8B365')

plt.title('encryption time comparison in different size')
plt.xlabel('image size')
plt.ylabel('Encryption time (second)')
plt.legend()

plt.tight_layout()
plt.show()
