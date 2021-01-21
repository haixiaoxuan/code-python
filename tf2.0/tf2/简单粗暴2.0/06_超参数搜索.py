import numpy as np
from tensorflow import keras
from sklearn.model_selection import RandomizedSearchCV

"""
    # 超参数搜索 RandomizedSearchCV
        1. 转换为 sklearn model   
        2. 定义超参数    
        3. 开始搜索
"""


def build_model(hidden_layer_num=1, layer_size=30, learning_rate=1e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(layer_size, activation="relu", input_shape=x_train.shape[1:]))
    for _ in range(hidden_layer_num-1):
        model.add(keras.layers.Dense(layer_size, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model


sklearn_model = keras.wrappers.scikit_learn.KerasRegressor(build_model)
sklearn_model.fit(x_train_scaled, y_train, epochs=10, validation_data=(x_valid_scaled, y_valid))



from scipy.stats import reciprocal
# f(x) = 1/(x*log(b/a))      a=<x<=b

param_distribute = {
    "hidden_layer_num": [1,2,3,4],
    "layer_size": np.arange(1, 100),
    "learning_rate": reciprocal.rvs(1e-4, 1e-2, size=10)
}


# 超参数搜索默认情况下会做交叉验证，把数据分为n份，然后取其中一份测试，可以修改参数 cv来改变，默认是3
# 表示在参数分布中找出十种组合，并行度为5
search = RandomizedSearchCV(sklearn_model, param_distribute, n_iter=10, n_jobs=1)
search.fit(x_train_scaled, y_train,
           epochs=10,
           validation_data=(x_valid_scaled, y_valid))




print(search.best_params_)
print(search.best_score_)
print(search.best_estimator_)
model = search.best_estimator_.model
model.evaluate(x_test_scaled, y_test)
















