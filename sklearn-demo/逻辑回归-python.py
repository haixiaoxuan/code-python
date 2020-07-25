#! -*-coding=utf8-*-

import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

x=[]
y=["xiaoxuan2","xiaoxuan1"]

# sklearn 的数据预处理
le = preprocessing.LabelEncoder()
le.fit(["xiaoxuan1", "xiaoxuan2"])
y = le.transform(y)     #[1,0]

# 管道机制 其中最后一步必须是估计器（Estimator），sc clf 为名称
lr = Pipeline([('sc', StandardScaler()),    # 规范化，使得各特征的均值为1 方差为0
               ('clf', LogisticRegression())])
lr.fit(x, y.ravel())
y_hat = lr.predict(x)
y_hat_prob = lr.predict_proba(x)





