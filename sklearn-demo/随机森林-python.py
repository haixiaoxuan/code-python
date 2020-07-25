# -*- coding:utf-8 -*-

import numpy as np
from sklearn.ensemble import RandomForestClassifier

def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]

# 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

if __name__ == "__main__":

    path = 'iris.data'  # 数据文件路径
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    x_prime, y = np.split(data, (4,), axis=1)

    # 随机森林  200 棵树组成的森林
    clf = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=3)
    rf_clf = clf.fit([],[])


