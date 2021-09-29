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

"""
    基于 sklearn 接口实现分类
"""

model = xgb.XGBClassifier(max_depth=10)



# watchlist = [(xg_train, 'train'), (xg_test, 'test')]
# num_round = 500
# bst = xgb.train(param, xg_train, num_round, watchlist, callbacks=[wandb.xgboost.wandb_callback()])

"""
    watchlist 可以实现边训练边评估
"""
