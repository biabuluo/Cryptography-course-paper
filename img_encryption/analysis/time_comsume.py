# -*-coding:utf-8 -*-

"""
# File       : time_comsume
# Time       ：2024/10/15 17:19
# Author     ：chenyu
# version    ：python 3.8
# Description：
"""
import time
import numpy as np
from BCIEA.experiment import encrypt as e1
from CDBEA.experiment import encrypt as e2
from teng.experiment import encrypt as e3
from pandas import DataFrame

# path = '../images/grey/cameraman.jpg'
record_1 = []
record_2 = []
record_3 = []

for i in range(60, 600, 10):
    print(i)
    start = time.time()
    H, W = i, i
    length = 4 * H * W
    a1 = np.random.randint(2, size=length)
    a2 = np.random.randint(2, size=length)
    _, _ = e1(a1, a2, i, i)
    end = time.time()
    record_1.append(end-start)

    start = time.time()
    H, W = i, i
    length = 4 * H * W
    a1 = np.random.randint(2, size=length)
    a2 = np.random.randint(2, size=length)
    _, _ = e2(a1, a2, H=i, W=i)
    end = time.time()
    record_2.append(end-start)

    start = time.time()
    H, W = i, i
    length = 4 * H * W
    a1 = np.random.randint(2, size=length)
    a2 = np.random.randint(2, size=length)
    _, _ = e3(a1, a2, i, i)
    end = time.time()
    record_3.append(end-start)

df = DataFrame({
    'iter': list(range(60, 600, 10)),
    'xu': record_1,
    'proposed': record_2,
    'teng': record_3
})

df.to_csv('./size_iter.csv', index=False)


