#! -*-coding=utf8-*-

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


x=[]
y=[]
# 如果指定 random_state ，则每次随机切分的数据是一样的
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
linreg = LinearRegression()
model = linreg.fit(x_train, y_train)

y_hat = linreg.predict(np.array(x_test))
# Mean Squared Error
mse = np.average((y_hat - np.array(y_test)) ** 2)
# Root Mean Squared Error
rmse = np.sqrt(mse)

##########################################################

from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV

model = Lasso()
# model = Ridge()
# 构造等比数列
alpha_can = np.logspace(-3, 2, 10)

# 人为指定参数， cv 表示交叉验证策略（即5折交叉验证）
lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
lasso_model.fit(x_train, y_train)

y_hat = lasso_model.predict(np.array(x_test))
mse = np.average((y_hat - np.array(y_test)) ** 2)
rmse = np.sqrt(mse)


############################################################

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('ss', StandardScaler()),
    # 线性回归的多项式深度为3
    ('poly', PolynomialFeatures(degree=3, include_bias=True)),
    # 构造特征，degree控制多项式的度，interaction_only： 默认为False，
    # 如果指定为True，那么就不会有特征自己和自己结合的项，二次项中没有a^2和b^2。
    ('linear', ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.99, 1], alphas=np.logspace(-3, 2, 5),
                            fit_intercept=False, max_iter=1e3, cv=3))
])

model.fit(x_train, y_train.ravel())
linear = model.get_params('linear')['linear']
# print u'系数：', linear.coef_.ravel()
y_pred = model.predict(x_test)
# R平方系数
r2 = model.score(x_test, y_test)
# 均方误差
mse = mean_squared_error(y_test, y_pred)






