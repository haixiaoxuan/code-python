import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 此行比较重要，画3D
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def test():
    x = np.random.random(100)
    y = np.random.random(100)
    color = np.random.random(100)
    size = np.random.random(100) * 100

    # marker 表示点的形状
    # alpha表示透明度
    plt.scatter(x, y, c=color, s=size, alpha=0.4, marker="o")

    # plt.colorbar()  # 显示色彩条状图
    plt.show()


def test2():
    # 三维散点图
    x = y = z = np.arange(10)
    size = np.random.random((10, 10, 10)) * 100  # shape=(10,10,10)
    d = np.meshgrid(x, y, z)  # shape=(3,10,10)

    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(d[1], d[0], d[2], c='r', s=size, marker='o', depthshade=True)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(u'三维', fontsize=20)
    plt.show()


def test3():
    """ 生成不同颜色的散点(相同label颜色相同) """
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 4, 5, 6])
    label = np.array([0, 0, -1, -1, 2])

    # 以label进行分类，label相同颜色相同
    plt.scatter(x, y, c=label)
    plt.show()


def test4():
    """ 第二种生成不同颜色的方案 """
    N = 50
    x = np.linspace(0., 10., N)
    y = np.sin(x) ** 2 + np.cos(x)

    # plt.figure()
    # label 为图右上角标签， 一般与 plt.legend() 连用
    plt.scatter(x, y, s=15, label=r"$y = sin^2(x) + cos(x)$")
    plt.legend()

    # 使得 x,y 轴比例尺相同
    plt.axis('equal')

    # 为坐标轴添加标签
    plt.xlabel(r'$x$ (rad)')
    plt.ylabel(r'$y$')

    # 更改图片尺寸
    plt.figure(figsize=(7, 7))

    from matplotlib.ticker import MultipleLocator
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(10))

    plt.show()


if __name__ == "__main__":
    test4()
