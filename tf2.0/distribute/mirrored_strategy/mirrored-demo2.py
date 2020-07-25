import tensorflow as tf

import numpy as np
import os

"""
    使用mirrored策略对自定义代码实现训练
"""


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# 向数组添加维度 -> 新的维度 == (28, 28, 1)
# 我们这样做是因为我们模型中的第一层是卷积层
# 而且它需要一个四维的输入 (批大小, 高, 宽, 通道).
# 批大小维度稍后将添加。
train_images = train_images[..., None]
test_images = test_images[..., None]


# 获取[0,1]范围内的图像。(归一化)
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)


strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
BUFFER_SIZE = len(train_images)
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = 10


# 将特征与label合并 创建Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)


# 转换为分布式Dataset
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)


# 创建模型
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
        ])
    return model


# 创建检查点目录以存储检查点。
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


# 定义损失函数 通常，在一台只有一个 GPU / CPU 的机器上，损失需要除去输入批量中的示例数
# 举一个例子，假设您有4个 GPU，批量大小为64. 输入的一个批次分布在各个副本（4个 GPU）上，每个副本获得的输入大小为16。
# 每个副本上的模型使用其各自的输入执行正向传递并计算损失。 现在，相较于将损耗除以其各自输入中的示例数（BATCH_SIZE_PER_REPLICA = 16），应将损失除以GLOBAL_BATCH_SIZE（64）
# 需要这样做是因为在每个副本上计算梯度之后，它们通过 summing 来使得在自身在各个副本之间同步。
with strategy.scope():
    # 将减少设置为“无”，以便我们可以在之后进行这个减少并除以全局批量大小。
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)
    # 或者使用 loss_fn = tf.keras.losses.sparse_categorical_crossentropy

    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)


    # 这些指标可以跟踪测试的损失，训练和测试的准确性。 您可以使用.result（）随时获取累积的统计信息。
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # 必须在`strategy.scope`下创建模型和优化器。
    model = create_model()
    optimizer = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)


    def train_step(inputs):
        images, labels = inputs

        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = compute_loss(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_accuracy.update_state(labels, predictions)
        return loss


    def test_step(inputs):
        images, labels = inputs

        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss.update_state(t_loss)
        test_accuracy.update_state(labels, predictions)


    # `experimental_run_v2`将复制提供的计算并使用分布式输入运行它。
    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.experimental_run_v2(train_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


    @tf.function
    def distributed_test_step(dataset_inputs):
        return strategy.experimental_run_v2(test_step, args=(dataset_inputs,))


    for epoch in range(EPOCHS):
        # 训练循环
        total_loss = 0.0
        num_batches = 0
        for x in train_dist_dataset:
            total_loss += distributed_train_step(x)
            num_batches += 1
        train_loss = total_loss / num_batches

        # 测试循环
        for x in test_dist_dataset:
            distributed_test_step(x)

        if epoch % 2 == 0:
            checkpoint.save(checkpoint_prefix)

        template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                    "Test Accuracy: {}")
        print(template.format(epoch + 1, train_loss,
                              train_accuracy.result() * 100, test_loss.result(),
                              test_accuracy.result() * 100))

        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()




