import tensorflow as tf


"""
    Keras Sequential/Functional API 模式建立模型
"""

# Sequential
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
])


# Functional API
inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dense(units=10)(x)
outputs = tf.keras.layers.Softmax()(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


# 当模型建立完成后，通过 tf.keras.Model 的 compile 方法配置训练过程：
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)


# 接下来，可以使用 tf.keras.Model 的 fit 方法训练模型：
# x ：训练数据；
# y ：目标数据（数据标签）；
# epochs ：将训练数据迭代多少遍；
# batch_size ：批次的大小；
# validation_data ：验证数据，可用于在训练过程中监控模型的性能
model.fit(data_loader.train_data, data_loader.train_label, epochs=num_epochs, batch_size=batch_size)


# 最后，使用 tf.keras.Model.evaluate 评估训练效果，提供测试数据及标签即可：
print(model.evaluate(data_loader.test_data, data_loader.test_label))


