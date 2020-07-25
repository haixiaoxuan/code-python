import matplotlib.pyplot as plt
import numpy as np


def height(x, y):
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)


x = np.linspace(-3, 3, 300)
y = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x, y)

# 为等高线填充颜色 10表示按照高度分成10层 alpha表示透明度 cmap表示渐变标准
plt.contourf(X, Y, height(X, Y), 10, alpha=0.75, cmap=plt.cm.hot)

# 使用contour绘制等高线, 可以添加 levels=[-1, -0.5, 0, 0.5, 1]，表示显示高度为这些值时的等高线
# 也可以指定这些等高线的参数，colors=list('kmrmk') linestyles=['--', '-.', '-', '-.', '--']
# linewidths=[1, 0.5, 1.5, 0.5, 1]
C = plt.contour(X, Y, height(X, Y), 10, colors='black')

# 在等高线处添加数字
plt.clabel(C, inline=True, fontsize=10)

# 去掉坐标轴刻度
plt.xticks(())
plt.yticks(())

plt.show()


