import tensorflow as tf
from tensorflow import keras

"""
    1. 自定义损失函数
    2. 自定义layer

"""




""" 
    自定义损失函数
"""


def customized_mse(y_true, y_hat):
    return tf.reduce_mean(tf.square(y_hat - y_true))


model = tf.keras.Sequential()
model.compile(loss=customized_mse)


class MeanSquaredError(tf.keras.losses.Loss):
    """ 或者 继承的方式实现 """
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))





"""
    自定义layer, 两种方式
        1. 继承
        2. lambda表达式方式
"""


class CustomizedDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = keras.layers.Activation(activation)
        super(CustomizedDenseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 构建所需参数
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[1], self.units),
                                      initializer='uniform', trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True)
        super(CustomizedDenseLayer, self).build(input_shape)

    def call(self, x):
        # 完整正向计算
        return self.activation(x @ self.kernel + self.bias)


# 通过lambda表达式方式, softplus 激活函数 -> log(1 + e^x)
customized_softplus = keras.layers.Lambda(lambda x: tf.nn.softplus(x))


model = tf.keras.models.Sequential([
    CustomizedDenseLayer(30, activation='relu', input_shape=[]),
    CustomizedDenseLayer(1),
    customized_softplus,
    # keras.layers.Dense(1,activation="softplus"),
    # keras.layers.Dense(1),keras.layers.Activation('softplus'),
])




"""
    自定义评估
        自定义评估指标需要继承 tf.keras.metrics.Metric 类，并重写 __init__ 、 update_state 和 result 三个方法。
        下面的示例对前面用到的 SparseCategoricalAccuracy 评估指标类做了一个简单的重实现：
"""


class SparseCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self):
        super().__init__()
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis=-1, output_type=tf.int32)), tf.int32)
        self.total.assign_add(tf.shape(y_true)[0])
        self.count.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.count / self.total





