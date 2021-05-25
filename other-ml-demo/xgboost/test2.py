import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_boston
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np


iris = load_iris()
x, y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

params = {
    "booster": "gbtree",
    "silent": 0,    # 0表示打印运行时信息， 1 表示不打印
    "num_feature": 4,   # boosting 过程使用的特征维数
    "objective": "multi:softmax",       # objective 用来定义学习任务以及对应的损失函数
    "num_class": 3,

    "gama": 0.1,    # 叶子节点进行划分时，需要损失函数减小的最小值
    "lambda": 2,    # 正则化权重
    "subsample": 0.7,   # 训练样本占总样本的比例， 防止过拟合
    "colsample_bytree": 0.7,    # 建立树时对特征进行采样的比例
    "min_child_weight": 3,  # 叶子节点继续划分的最小样本权重和
    "eta": 0.1      # 加法模型中的收缩步长
}
plst = params.items()

# 数据格式转换
dTrain = xgb.DMatrix(x_train, y_train)
dTest = xgb.DMatrix(x_test, y_test)

# 迭代次数，对于分类问题，就是每个类别的迭代次数， 所以基学习器的总个数 = 迭代次数 * 类别个数
num_rounds = 50
model = xgb.train(params, dTrain, num_rounds)

y_pred = model.predict(dTest)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# 显示特征重要度
plot_importance(model)
plt.show()

# 可视化树的生成情况, num_trees 是树的索引，索引是从0开始
plot_tree(model, num_trees=5)

model.dump_model("model.txt")

