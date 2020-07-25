import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 数据
a = np.array([22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27])

# 这里也可以切分多维数组，只需要修改bins，（切分三维，bins=[5,5,5]）
# hist, bins = np.histogram(a, bins=[0, 20, 40, 60, 80, 100]) # 将数据分为五段
hist, bins = np.histogram(a, bins=5)    # 将数据分为五段（平均）

print(hist)     # 每一段所拥有的元素个数
print(bins)     # 分段标准











