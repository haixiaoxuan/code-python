import matplotlib.pyplot as plt
import numpy as np

x = range(10)
y = range(10)

I = np.random.random((10, 10))/255
print(I)
# cmap=plt.cm.gray_r 指定颜色为灰色
# plt.imshow(I, cmap=plt.cm.gray_r, interpolation='nearest')   # I 是矩阵
plt.matshow(I)

plt.savefig("hh.png")

plt.show()
