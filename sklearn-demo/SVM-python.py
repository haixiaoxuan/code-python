import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np

"""

    kernel      rbf|linear      高斯核|线性核
    C           惩罚参数， 默认是1
    degree      多项式的维度，默认是3
    max_iter
    tol         停止训练的最小误差】
    decision_function_shape    ovo|ovr

"""


def cal_accuracy(a, b):
    """ 计算准确率 """
    acc = a.values.ravel() == b.ravel()
    print('正确率：', np.mean(acc))


path = "data/10.iris.data"
df = pd.read_csv(path, header=None)

x = df.drop(4, axis=1)
y = pd.DataFrame({"label": df[4]})
y[y["label"] == "Iris-setosa"] = 0
y[y["label"] == "Iris-versicolor"] = 1
y[y["label"] == "Iris-virginica"] = 2

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

# 构建模型
clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(x_train, y_train.values.ravel())

print("准确率：" + str(clf.score(x_train, y_train)))
cal_accuracy(y_train, clf.predict(x_train))

print("测试集准确率：" + str(clf.score(x_test, y_test)))
cal_accuracy(y_test, clf.predict(x_test))

print("样本到决策面的距离 " + str(clf.decision_function(x_train)))  # 与decision_function_shape取 ovr 、ovo 有关

print('支撑向量的数目：', clf.n_support_)
print('支撑向量的系数：', clf.dual_coef_)
print('支撑向量：', clf.support_)

""" ******************************************* """

# 回归
# kernel    rbf|linear|poly
svr_rbf = svm.SVR(kernel='rbf', gamma=0.2, C=100)
