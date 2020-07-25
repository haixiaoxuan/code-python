import tensorflow as tf
from tensorflow import keras


"""
    mirroredStrategy:
        note: 这种策略会在每个GPU上保存一份模型参数，然后计算完成后同步更新。
        可以指定设备 strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
        strategy.num_replicas_in_sync       可以获取设备数量

    keras_model:
        1. 将 batch_size 设置为 batch_size * num_gpu
        2. strategy = tf.distribute.MirroredStrategy()
           with strategy.scope():
               model = ...
               model.compile...

    estimator_model:
        strategy = tf.distribute.MirroredStrategy()
        config = tf.estimator.RunConfig(train_distribute=strategy)
        estimator = keras.estimator.model_to_estimator(model, config=config)
        estimator.train()
        


"""


# 自定义训练流程实现分布式


# 构造dataset
def make_dataset(images, labels, epochs, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size).prefetch(50)
    return dataset


epochs = 1
batch_size = 128
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    train_dataset = make_dataset(x_train_scaled, y_train, epochs, batch_size)
    valid_dataset = make_dataset(x_valid_scaled, y_valid, epochs, batch_size)

    # 做数据分发
    train_dataset_distribute = strategy.experimental_distribute_dataset(train_dataset)
    valid_dataset_distribute = strategy.experimental_distribute_dataset(valid_dataset)


with strategy.scope():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="selu", input_shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="selu"))
    model.add(keras.layers.MaxPool2D(pool_size=2))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="selu"))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="selu"))
    model.add(keras.layers.MaxPool2D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="selu"))
    model.add(keras.layers.Dense(10, activation="softmax"))


"""
    自定义训练流程：
        1. 定义损失函数
        2. 定义训练函数，获取准确率和loss
        3. 定义验证函数
        4. 定义循环
"""

with strategy.scope():
    # 定义损失函数
    # reduction 训练一个样本获得损失值之后如何聚合
    # SUM_OVER_BATCH_SIZE 这种策略只适合单机单卡情况
    # loss_func = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)


    def compute_loss(labels, predicts):
        # 如果是单机单卡不用本层封装
        pre_replica_loss = loss_func(labels, predicts)
        return tf.nn.compute_average_loss(pre_replica_loss, global_batch_size=batch_size)


    # 定义训练集测试集上的指标统计(这里的是累加的变量)
    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    # 定义train_step
    optimizer = tf.keras.optimizers.SGD(lr=0.01)


    def train_step(inputs):
        images, labels = inputs
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            # loss = loss_func(labels, predictions) 单机
            loss = compute_loss(labels, predictions)
        gradient = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))
        train_accuracy.update_state(labels, predictions)
        return loss


    @tf.function
    def distribute_train_step(inputs):
        pre_replica_loss = strategy.experimental_run_v2(train_step, args=(inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, pre_replica_loss, axis=None)


    def test_step(inputs):
        images, labels = inputs
        predictions = model(images, training=True)
        # loss = loss_func(labels, predictions)
        loss = compute_loss(labels, predictions)
        test_loss.update_state(loss)
        test_accuracy.update_state(labels, predictions)


    @tf.function
    def distribute_test_step(inputs):
        strategy.experimental_run_v2(train_step, args=(inputs,))


    epochs = 10
    for epoch in range(epochs):
        for x in train_dataset:
            # total_loss = train_step(x)
            total_loss = distribute_train_step(x)
            print(total_loss)

        for x_valid in valid_dataset:
            # test_step(x_valid)
            distribute_test_step(x_valid)
        test_loss.reset_states()
        test_accuracy.reset_states()
        train_accuracy.reset_states()


