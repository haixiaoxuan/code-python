# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pydotplus

"""
    criterion                   划分标准   gini | entropy                   
    splitter                    拆分策略   
    max_depth                   最大深度
    min_samples_split           最小划分样本数
    min_samples_leaf            叶子节点最小样本数
    min_weight_fraction_leaf    叶子节点最小权重
    max_features                划分时考虑的最大特征数量
    random_state
    max_leaf_nodes              最大叶节点的数量
    min_impurity_decrease
    min_impurity_split
    class_weight                标签权重``{class_label: weight}``
    presort                     预分类
    
    
"""

def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]

# 花萼长度、花萼宽度，花瓣长度，花瓣宽度
iris_feature_E = 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    path = 'iris.data'  # 数据文件路径
    # converters 将第四列按照 iris_type 函数转换
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    x, y = np.split(data, (4,), axis=1)
    # 数据集拆分
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)

    # 决策树参数估计
    # min_samples_split = 10：如果该结点包含的样本数目大于10，则(有可能)对其分支
    # min_samples_leaf = 10：若将某结点分支后，得到的每个子结点样本数目都大于10，则完成分支；否则，不进行分支
    model = DecisionTreeClassifier(criterion='entropy', max_depth=6)
    model = model.fit(x_train, y_train)
    y_test_hat = model.predict(x_test)      # 测试数据
    
    # 1 - 输出 .dot 文件
    # with open('iris.dot', 'w') as f:
    #     tree.export_graphviz(model, out_file=f)

    # 将结果图写成 pdf 格式
    dot_data = tree.export_graphviz(model, out_file=None, feature_names=iris_feature_E, class_names=iris_class,
                                    filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_pdf('iris.pdf')
    # 将结果图写成 png 格式
    f = open('iris.png', 'wb')
    f.write(graph.create_png())
    f.close()









