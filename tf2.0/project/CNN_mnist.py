import tensorflow as tf
from MLP_mnist import MnistDataLoader


class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,             # 卷积层神经元（卷积核）数目
            kernel_size=[5, 5],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            activation=tf.nn.relu   # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)                  # [batch_size, 28, 28, 32]
        x = self.pool1(x)                       # [batch_size, 14, 14, 32]
        x = self.conv2(x)                       # [batch_size, 14, 14, 64]
        x = self.pool2(x)                       # [batch_size, 7, 7, 64]
        x = self.flatten(x)                     # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)                      # [batch_size, 1024]
        x = self.dense2(x)                      # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


if __name__ == "__main__":
    train_path = "/root/xiaoxuan/data/mnist/train.csv"
    test_path = "/root/xiaoxuan/data/mnist/test.csv"
    loader = MnistDataLoader(train_path, test_path)
    print("数据加载完成...")

    batch_size = 128
    learning_rate = 0.001

    model = CNN()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for step in range(50000):
        X, y = loader.get_batch(batch_size)
        X = X.reshape(-1, 28, 28, 1)

        with tf.GradientTape() as tape:
            y_pred = model(X)

            # categorical_crossentropy | sparse_categorical_crossentropy
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

        if step % 100 == 0:
            # 模型评估
            test_y_ = model(loader.test_x.reshape(-1, 28, 28, 1))
            sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            sparse_categorical_accuracy.update_state(y_true=loader.test_y, y_pred=test_y_)
            print("step: %d" % step, "test accuracy: %f" % sparse_categorical_accuracy.result(), "loss: %f" % loss)





