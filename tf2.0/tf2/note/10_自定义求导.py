import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

""" 
    自定义求导
"""


def g(x1, x2):
    return (x1 + 5) * (x2 ** 2)


def f(x):
    return 3. * x ** 2 + 2. * x - 1


x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
with tf.GradientTape() as tape:
    z = g(x1, x2)
# 默认情况下，tape对象只能调一次就会被删除，不能再用
dz_x1 = tape.gradient(z, x1)        # g对x1求偏导
print(dz_x1)


try:
    dz_x2 = tape.gradient(z, x2)       # 调用第二次会报错
except RuntimeError as ex:
    print(ex)


# 将参数设为True可以调多次，但是要手动删除
with tf.GradientTape(persistent=True) as tape:
    z = g(x1, x2)

dz_1 = tape.gradient(z, x1)
dz_2 = tape.gradient(z, x2)
print(dz_1, dz_2)
del tape


# 一次求出关于两个变量的偏导
with tf.GradientTape() as tape:
    z = g(x1, x2)
dz_x1x2 = tape.gradient(z, [x1, x2])
print(dz_x1x2)


# 对常量求偏导
x1_constant = tf.constant(2.0)
x2_constant = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x1_constant)  # 对常量求偏导需要设置关注
    tape.watch(x1_constant)
    z = g(x1_constant, x1_constant)
dz_x1x2 = tape.gradient(z, [x1_constant, x1_constant])
print(dz_x1x2)


# 一个变量对两个目标函数求偏导
x = tf.Variable(5.0)
with tf.GradientTape() as tape:
    z1 = 3 * x
    z2 = x ** 2
tape.gradient([z1, z2], x)


# 求二阶导
with tf.GradientTape(persistent=True) as outer_tape:
    with tf.GradientTape(persistent=True) as inner_tape:
        z = g(x1, x2)
    inner_grads = inner_tape.gradient(z, [x1, x2])
outer_grads = [outer_tape.gradient(inner_grad, [x1, x2])
               for inner_grad in inner_grads]

print(outer_grads)
del inner_tape
del outer_tape


# 实现梯度下降
learning_rate = 0.1
x = tf.Variable(0.0)
for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)
    x.assign_sub(learning_rate * dz_dx)
print(x)


# 将自定义求导与优化器结合使用
optimizer = keras.optimizers.SGD(lr=learning_rate)
for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)
    optimizer.apply_gradients([(dz_dx, x)])
print(x)



"""
    自动求导，应用案例
"""


housing = fetch_california_housing()
x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state=7)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state=11)
print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)


# metric 使用
metric = keras.metrics.MeanSquaredError()
print(metric([5.], [2.]))  # 当成函数一样使用，输入两个列表
print(metric([0.], [1.]))
print(metric.result())  # 具有累加功能

metric.reset_states()  # 清空状态，即不累加
metric([1.], [3.])
print(metric.result())


# 1、batch遍历训练集 metric
#  1.1、自动求导
# 2、epoch结束验证集metric

epochs = 100
batch_size = 32
steps_pre_epoch = len(x_train_scaled)
optimizer = keras.optimizers.SGD()
metric = keras.metrics.MeanSquaredError()


# 返回一个批次数据
def random_batch(x, y, batch_size=32):
    idx = np.random.randint(0, len(x), size=batch_size)
    return x[idx], y[idx]


model = tf.keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=x_train.shape[1:]),
    keras.layers.Dense(1)
])

for epoch in range(epochs):
    metric.reset_states()
    for step in range(steps_pre_epoch):
        x_batch, y_batch = random_batch(x_train_scaled, y_train, batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            loss = tf.reduce_mean(keras.losses.mean_squared_error(y_batch, y_pred))
            metric(y_batch, y_pred)
        grads = tape.gradient(loss, model.variables)
        grads_and_vars = zip(grads, model.variables)
        optimizer.apply_gradient(grads_and_vars)
        print("\rEpoch", epoch, "train_mse:", metric.result().numpy(), end="")

    y_valid_pred = model(x_valid_scaled)
    vaild_loss = tf.reduce_mean(keras.losses.mean_squared_error(y_valid_pred, y_valid))
    print("\t", "valid mse: ", vaild_loss.numpy())

model.compile(loss="mean_squared_error", optimizer='sgd')
callbacks = [keras.callbacks.EarlyStopping(patience=5,min_delta=1e-2)]

