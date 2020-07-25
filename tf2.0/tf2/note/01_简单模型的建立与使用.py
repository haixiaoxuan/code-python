import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
import os
from tensorflow import keras

"""
    使用模型训练之前可以先对数据进行归一化, 可以使用 sklearn等
    
    loss：
        sparse_categorical_crossentropy
        categorical_crossentropy        y值已经经过one-hot编码，使用这种损失函数
        mean_squared_error
"""


# 使用sequential 定义模型
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))


# 如果y已经经过one-hot编码，则直接使用 categorical_crossentropy 损失函数
# metrics 表示要统计什么指标, 除了统计 accuracy还会统计loss
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])


# 模型属性
model.layers
model.summary()
model.metrics_names



# 每训练完一个epoch就会在验证集上做一个验证
# validation_split=0.2  可以不使用validation_data，直接在训练数据集上切分验证集
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid))

type(history)       # tensorflow.python.keras.callbacks.History


# 每个 epochs 会输出四个值
history.history
{'loss': [2.3026957498030227, ],
 'accuracy': [0.09767273, ],
 'val_loss': [2.3026871742248534, ],
 'val_accuracy': [0.0914, ]}


def draw_res(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


draw_res(history)


# 在测试集测试
model.evaluate(x_test, y_test)
# 可以使用模型的此属性来知道测试返回的指标是什么
model.metrics_names





