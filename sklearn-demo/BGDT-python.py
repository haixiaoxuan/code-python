import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.model_selection import GridSearchCV

"""
    https://www.cnblogs.com/pinard/p/6143927.html
    
    random_state
    
    n_estimators        (default=100)   
    loss : {'deviance',                 对数似然损失函数（等同于逻辑回归，用概率输出分类）
            'exponential'}              指数损失函数（adaboost 算法）,不能用于多分类
    learning_rate       (default=0.1)   每个若学习器的权重系数
    subsample                           子采样（不放回抽样）
    init                                初始化f0   默认 None
    
    max_features        (default=None)  划分时考虑特征数
    max_depth           (default=3)     决策树最大深度
    max_leaf_nodes                      最大的叶节点的数量
    criterion           (default="friedman_mse")    mse|mae     决策树节点划分标准
    min_samples_split   (default=2)     节点划分的最小样本数，如果为分数表示百分比
    min_samples_leaf    (default=1)     叶子节点最小样本数，如果小于此值，则会和兄弟节点一起被剪枝
    min_weight_fraction_leaf            叶子节点所有样本权重和的最小值，如果小于此值，则剪枝
    min_impurity_split  (default=1e-7)  节点划分最小不纯度，如果小于此值，则不再划分
    min_impurity_decrease               节点划分的最小不纯度减少量，如果采用此划分，可以使不纯度的减少量大于此值，则采用此划分
    
    verbose     (default=0)             如果为1，每隔一段时间打印进度，如果大于1，则每棵树打印一下
    validation_fraction                 预留用于验证集的比例，用于提前停止，只有当n_iter_no_change为整数时，才启用
    n_iter_no_change                    用于判断当验证分数没有提高时，提前终止训练。会使用validation_fraction预留的验证集进行计算
    tol                                 n_iter_no_change 次迭代，loss值的减少没有大于tol，则提前终止训练
    
    warm_start                  是否复用上一次结果，例：如果复用，则每次循环只会重新生成一颗数
                                for i in range(99):
                                    classifier.n_estimators += 1
                                    classifier.fit(X_train, y_train)
    presort          (default="auto")   是否进行预分类   
    
"""

df = pd.read_csv("E:\\project\\python-test\\test1\\10.iris.data")

label = df["label"].values
feature = df.drop(columns="label").values

model = GradientBoostingClassifier(n_estimators=5)
model.fit(feature, label)

predict = model.predict(feature)  # 预测结果
pre_prob = model.predict_proba(feature)  # 概率分布 softmax类似

accuracy = metrics.accuracy_score(label, predict)  # 准确率计算


# AUC = metrics.roc_auc_score(label, pre_prob)        # AUC 不支持多分类


def test4():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn import metrics
    import time

    start = time.time()
    inputpath1 = 'E:\\data\\data_cell_lable_0521_rsrp_five3_all.csv'
    df_data = pd.read_csv(inputpath1)
    df_data = df_data.dropna(axis=0, how='any')

    x1 = df_data.drop(['label'], axis=1)
    y1 = df_data['label']

    train_x, test_x, train_y, test_y = train_test_split(x1, y1, train_size=0.8, test_size=0.2)

    gbdt = GradientBoostingClassifier(n_estimators=1, verbose=2, warm_start=True)

    for i in range(100):
        gbdt.fit(train_x, train_y)
        y_predict = gbdt.predict(test_x)
        y_train_predict = gbdt.predict(train_x)
        accuracy = metrics.accuracy_score(test_y, y_predict)
        train_accuracy = metrics.accuracy_score(train_y, y_train_predict)
        print(str(gbdt.n_estimators) + "  验证准确率：" + str(accuracy) + "  训练准确率： " + str(train_accuracy))

        gbdt.n_estimators += 1

    end = time.time()
    print("总耗时 ：%.2f s" % (end - start))
