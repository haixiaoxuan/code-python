import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

x = range(10)
plt.subplot(2, 1, 1)  # 两行一列第一个
plt.plot(x, np.sin(x), "o")  # o 表示散点图
print(plt)
plt.xticks(size=10, rotation=30)
# 去掉x轴刻度
# plt.xticks(())
# plt.yticks(())

plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))
plt.ylim(-5, 1)  # 对y的范围做限制

plt.tight_layout(1.4) # 会自动调整子图参数，使之填充整个图像区域
plt.show()

figure = plt.figure()
plt.subplots()