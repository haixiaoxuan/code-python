import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpt
import seaborn as sns


train_path = r"C:\Users\xiexiaoxuan\Desktop\腾讯广告算法\2017腾讯赛数据集\final\train.csv"
test_path = r"C:\Users\xiexiaoxuan\Desktop\腾讯广告算法\2017腾讯赛数据集\final\test.csv"

# df = pd.read_csv(train_path, index_col=None)
# df.head()
# df.size
#
# # 检测是否有数据缺失
# df.isnull().any()
#
# df.drop("conversionTime", axis=1)
# df.take()
# df.sample()
#
# df.size()
# df.__sizeof__()

import os
print(os.environ["path"])

from sklearn.ensemble import AdaBoostClassifier




