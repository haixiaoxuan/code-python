import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# 显示中文
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

x1_min, x1_max = 0, 10
x2_min, x2_max = 0, 10
# 生成网格采样点 200*200
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]

# 所有的点应用于模型，得出每个点的分类情况 grid_hat
grid_test = np.stack((x1.flat, x2.flat), axis=1)
grid_hat = [0 if a > b else 1 for a, b in grid_test]
grid_hat = np.array(grid_hat).reshape(x1.shape)

# 网格点的颜色 #77E0A0', '#FF8080', '#A0A0FF' '#A0FFA0', '#FFA0A0', '#A0A0FF'
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
# 样本点的颜色
cm_dark = mpl.colors.ListedColormap(['g', 'r'])

# 在图像上绘制网格点
plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
# 绘制样本点，degecolors表示边缘为黑色，s表示大小
plt.scatter([1, 2, 3, 4], [4, 3, 2, 1], c=[0, 0, 1, 1], edgecolors='k', s=50, cmap=cm_dark)  # 样本
# 圈中测试集样本
plt.scatter([4], [1], s=120, facecolors='none', zorder=10)

plt.xlabel("x", fontsize=13)
plt.ylabel("y", fontsize=13)

plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u"散点分类图", fontsize=15)
plt.grid()
plt.show()
plt.savefig("/tmp/7777777777777777777777.png")




# N, M = 500, 500  # 横纵坐标各生成五百个点
# x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
# x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
# t1 = np.linspace(x1_min, x1_max, N)
# t2 = np.linspace(x2_min, x2_max, M)
# x1, x2 = np.meshgrid(t1, t2)
