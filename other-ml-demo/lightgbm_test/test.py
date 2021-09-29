import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

breast = load_breast_cancer()
x, y = breast.data, breast.target
feature_name = breast.feature_names

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

train_data = lgb.Dataset(x_train, y_train)
test_data = lgb.Dataset(x_test, y_test, reference=train_data)       # 用来做验证集

boost_round = 50
early_stop_round = 10   # 验证集如果在10轮中没有提高，则提前停止

params = {
    "boosting_type": "gbdt",
    "objective": "regression",  # 目标函数
    "metric": {"l2", "auc"},    # 评估函数
    "num_leaves": 31,           # 叶子节点个数,   一般设为 2^(max_depth
    # )
    "learning_rate": 0.05,      # 学习率
    "feature_fraction": 0.9,    # 建树时特征选择比例
    "bagging_fraction": 0.8,    # 建树的样本采样比例，
    "bagging_freq": 5,          # 意味着每k次迭代执行 bagging
    "verbose": 1
}

results = {}
model = lgb.train(params, train_data,
                  num_boost_round=boost_round,
                  valid_sets=(test_data, train_data),
                  early_stopping_rounds=early_stop_round,
                  evals_result=results)
print(results)

pred = model.predict(x_test, num_iteration=model.best_iteration)
print(pred)

lgb.plot_metric(results)
plt.show()

lgb.plot_importance(model, importance_type="split")
plt.show()