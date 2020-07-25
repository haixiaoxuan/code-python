import matplotlib.pyplot as plt
import matplotlib as mpl

# 显示中文
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

x = range(10)
y = x
# 绘制带顶点的折线图
plt.plot(x, y, "r-", x, y, "go", linewidth=2, markersize=8)

# 调节边距(图片到画框的边距)
plt.tight_layout(30)

# 调节标题占比
plt.title("abc")
plt.subplots_adjust(top=0.5)

# 对折线图中的顶点进行标注
for a, b in zip(x, y):
    plt.text(a, b+0.5, "%.2f" % b, ha="center", va="bottom", fontsize=10)

# 如果一个图中有多条线
plt.legend(loc=1)  # 添加标签，loc表示指定标签位置,1表示右上角


plt.show()

"""
marker
    .   point
    ,   pixel
    o   circle
    v   下三角形
    ^   上三角形
    <   左三角形

color
    b：  blue
    g:  green
    r:  red
    c:  cyan
    m:  magenta
    y:  yellow
    k:  black
    w:  white

linestyle
    - or solid 粗线
    -- or dashed dashed line
    -. or dashdot dash-dotted
    : or dotted dotted line
    'None' draw nothing
    ' ' or '' 什么也不绘画
"""
