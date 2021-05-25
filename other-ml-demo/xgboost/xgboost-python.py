import xgboost as xgb
import numpy as np

"""
    如果在运行 spark-xgboost 一直报 19/07/10 13:18:49 WARN amfilter.AmIpFilter: Could not find proxy-user cookie, so user will not be set
        可能是由于 worker数过小所导致的，申请不到足够的资源，调大即可
        
    https://xgboost.readthedocs.io/en/latest/python/python_intro.html
    不支持类别编码，如果需要对类别进行处理，在之前进行one-hot编码
    
    使用 xgboost4j-spark 特别注意版本问题，0.81版本有bug，如果特征数过多会导致程序异常挂掉
    
    参数类型：
        General parameters 
        Booster parameters 
        Learning task parameters
        
    General parameters
        booster         默认是gbtree，gbtree和dart都是树模型(gbtree | gblinear | dart) 
        verbosity       打印消息的详细程度，默认是1  (0(静默)|1(警告)|2(info)|3(debug))
        nthread         默认为最大线程数
        disable_default_eval_metric     禁用默认度量指标，默认是0，>0表示开启禁用
        
    Tree Booster
        eta             范围[0,1]，每一步提升的权重，default=0.3
        gamma           范围[0,∞]，在树的叶节点上进行进一步分区所需的最小损耗减少，default=0
        max_depth       数的深度[0,∞]
        min_child_weight    范围[0,∞]，子节点的最小权重和，小于此值则不split，default=1
        subsample       范围(0,1]，训练样本比例，default=1
        lambda          L2正则项的权重，default=1
        alpha           L1正则项，default=0
        tree_method     数的构造算法(auto|exact|approx|hist|gpu_hist)
        scale_pos_weight    控制正负样本的权重平衡，default=1
        
    Learning task parameters
        objective       这是分类目标，常用(reg:logistic | binary:logistic | binary:logitraw | multi:softmax | "multi:softprob")
        base_score      所有实例的初始得分，如果迭代次数过大，则不会有太大影响
        eval_metric     评估指标(mse|mae|...)
        seed
    

"""


def log_reg(y_hat, y):
    """ 自定义损失函数的梯度和二阶导 """
    p = 1.0 / (1.0 + np.exp(-y_hat))
    g = p - y.get_label()
    h = p * (1.0-p)
    return g, h


def error_rate(y_hat, y):
    """ 自定义 错误率 """
    return 'error', float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)


# 读取数据 ，数据格式 libsvm
data_train = xgb.DMatrix('14.agaricus_train.txt')
data_test = xgb.DMatrix('14.agaricus_test.txt')


# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=50)
# data_train = xgb.DMatrix(x_train, label=y_train)
# data_test = xgb.DMatrix(x_test, label=y_test)


# binary:logistic/logitraw
param = {'max_depth': 2,
         'eta': 1,
         'silent': 1,
         'objective': 'binary:logistic'     # 使用 lr做 二分类
         }

watchlist = [(data_test, 'eval'), (data_train, 'train')]

n_round = 3         # 表示树的颗数
bst = xgb.train(param,
                data_train,
                num_boost_round=n_round,
                evals=watchlist,
                obj=log_reg,
                feval=error_rate)

bst.dump_model('dump.raw.txt')
bst.dump_model('dump.raw.txt', 'featureMap')

y_hat = bst.predict(data_test)
y = data_test.get_label()
error = sum(y != (y_hat > 0.5))
error_rate = float(error) / len(y_hat)
print('样本总数：\t', len(y_hat))
print('错误数目：\t%4d' % error)
print('错误率：\t%.5f%%' % (100 * error_rate))

# ===========================================================

from xgboost.sklearn import XGBClassifier

...

"""
https://blog.csdn.net/cy_tec/article/details/80209453
    feature map 文件:
    0   cap-shape=bell  i
    1   cap-shape=conical   i
    2   cap-shape=convex    i
    3   cap-shape=flat  i
    .
    .
    .
    通过实例我们可以看出，feature map 的格式：
        <特征 id> <特征名称=特征值> <q or i or int>
        - 特征 id 排序从 0 开始升序
        - i 二选一的特征
        - q 数量值，如年龄、时间。这个值可以是空
        - int 整型特征，它的决策边界也应该是整型
        
https://github.com/Far0n/xgbfi
    使用xgbfi进行特征重要度分析

"""

# ======================================================

import xgboost as xgb
import pandas as pd

inputpath1 = '/home/etluser/xiexiaoxuan/xiexiaoxuan-test/xgboost/train3'
df_data = pd.read_csv(inputpath1)
df_data = df_data.dropna(axis=0, how='any')

x1 = df_data.drop(['label'], axis=1)
y1 = df_data['label']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x1, y1, train_size=0.8, random_state=1)
data_train = xgb.DMatrix(x_train,label = y_train.values)
data_test = xgb.DMatrix(x_test,label = y_test.values)

param = {'max_depth': 10,
         'objective': 'multi:softprob',
         "num_class": 10}
watchlist = [(data_test, 'eval'), (data_train, 'train')]
bst = xgb.train(param,data_train, num_boost_round=10, evals=watchlist)

y_hat = bst.predict(data_test)
y = data_test.get_label()
from sklearn import metrics
metrics.accuracy_score(y,y_hat)