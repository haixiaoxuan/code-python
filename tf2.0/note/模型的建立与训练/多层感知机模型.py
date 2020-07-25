import tensorflow as tf
import numpy as np

"""
    多层感知机 Multilayer Perceptron, MLP
    
    note：
        1. 交叉熵损失函数
            tf.keras.losses.categorical_crossentropy
            tf.keras.losses.sparse_categorical_crossentropy 
            后者真是的标签y_true可以直接传入int类型的值，而前者需要通过 one-hot 编码，生成与y_pre 一样的概率矩阵
"""


class MNISTLoader():
    """ 加载 mnist 数据集"""
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)      # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)        # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)    # [60000]
        self.test_label = self.test_label.astype(np.int32)      # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]


class MLP(tf.keras.Model):
    """ 全连接神经网络 """
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()    # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):         # [batch_size, 28, 28, 1]
        x = self.flatten(inputs)    # [batch_size, 784]
        x = self.dense1(x)          # [batch_size, 100]
        x = self.dense2(x)          # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


if __name__ == "__main__":
    num_epochs = 5
    batch_size = 50
    learning_rate = 0.001

    model = MLP()
    data_loader = MNISTLoader()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)

            # 交叉熵损失函数
            # categorical_crossentropy | sparse_categorical_crossentropy
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

        # 模型评估
        sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        num_batches = int(data_loader.num_test_data // batch_size)
        for batch_index in range(num_batches):
            start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
            y_pred = model.predict(data_loader.test_data[start_index: end_index])
            sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index],
                                                     y_pred=y_pred)
        print("test accuracy: %f" % sparse_categorical_accuracy.result())






