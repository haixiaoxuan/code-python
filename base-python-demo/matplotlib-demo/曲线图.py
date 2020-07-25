import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# 显示中文
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

x = np.linspace(1, 10, 200)
y = np.sin(x)
z = np.cos(x)

fig = plt.figure(figsize=(3, 4), facecolor="w")
plt.plot(x, y, label="正弦函数", color="red", linewidth=1)
plt.plot(x, z, "b--", label="$cos(x)$")  # 第三个参数表示曲线类型，b加粗 --虚线

# 画折线图时把顶点画出来
# plt.plot(x, y, "r-", x, y, "go", linewidth=2, markersize=8)


plt.title("正弦函数")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc=1)  # 添加标签，loc表示指定标签位置,1表示右上角

# plt.grid(True)  # 显示方格
# fig.savefig("hh.png") # 保存为图片
# plt.savefig("xx.png")

# xticks 表示对x轴刻度进行标注，rotation表示旋转 30度
plt.xticks([3, 6, 9], ["a", "b", "c"], size=10, rotation=30)

plt.show()
