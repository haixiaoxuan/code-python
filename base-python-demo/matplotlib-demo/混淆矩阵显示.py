import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
import numpy as np

# 混淆矩阵
matrix = [[113, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 108, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 93, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 115, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 88, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 80, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 107, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 101, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 89, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 106]]

plt.figure(figsize=(5, 5))

cdict = {
    'red': ((0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
    'green': ((0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
    'blue': ((0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))
}
cm = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)


# spring summer autumn winter
# plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.summer)

# Blues_r 表示反着表示
# https://matplotlib.org/examples/color/colormaps_reference.html
plt.imshow(matrix, interpolation="nearest", cmap=plt.get_cmap('Blues'))
plt.title("confusion_matrix")


# 添加颜色渐变条
plt.colorbar()

plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ["a", "b", "c", "d", "e", "f", "g", "h", "m", "n"], rotation=0)
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ["a", "b", "c", "d", "e", "f", "g", "h", "m", "n"], rotation=0)

for i, j in itertools.product(range(len(matrix)), range(len(matrix))):
    plt.text(j, i, matrix[i][j], horizontalalignment="center",
             color="black")

# 自动调整
plt.tight_layout()
plt.xlabel("True Label")
plt.ylabel("Predicted Label")

# 调节图片边距
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
# plt.savefig("hh.png")
plt.show()