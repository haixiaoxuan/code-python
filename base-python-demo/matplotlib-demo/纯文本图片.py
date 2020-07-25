text = """
             precision    recall  f1-score   support 

          7       0.71      0.30      0.42      1169 
          3       0.44      0.41      0.43      1179
          4       0.49      0.43      0.46      1033
          6       0.60      0.46      0.52      1079
          1       0.45      0.93      0.61      1231
          8       0.26      0.27      0.26      1072
          0       0.55      0.57      0.56      1086
          九       0.32      0.55      0.41      1089
          2       0.54      0.24      0.33      1096
          5       0.56      0.31      0.40       966

        avg / total       0.49      0.45      0.44     11000

"""

# text = "hello"
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


fig = plt.figure(figsize=(5, 5))

plt.axis([0, 2, 0, 2])
plt.text(1, 1, text, ha="center", va='center', fontsize=12, wrap=True)

# 去掉坐标轴
plt.axis('off')
plt.xticks([])
plt.yticks([])

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.close()
plt.show()

