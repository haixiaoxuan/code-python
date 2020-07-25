import tensorflow as tf
import pandas as pd


"""
    使用多层感知机来处理 mnist 数据集
"""


class MnistDataLoader():
    """ 加载 mnist 数据集 """

    def __init__(self, train, test):
        train_df = pd.read_csv(train)
        test_df = pd.read_csv(test)
        self.train_x, self.train_y = self._etl(train_df)
        self.test_x, self.test_y = self._etl(test_df)
        # 初始化索引位置
        self.batch_index = 0
        self.data_length = len(self.train_y)
        self.test_x = self.test_x.values
        self.test_y = self.test_y.values

    def _etl(self, df):
        """ 特征标签分离"""
        x = df.drop("label", axis=1)
        y = df["label"]
        return x, y

    def get_batch(self, batch_size):

        if self.batch_index + batch_size > self.data_length:
            self.batch_index = 0

        current_index = self.batch_index
        self.batch_index += batch_size
        return self.train_x.iloc[current_index: self.batch_index].values, \
               self.train_y.iloc[current_index: self.batch_index].values


class MLP(tf.keras.Model):
    """ 全连接神经网络 """
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()    # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=1000, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):         # [batch_size, 784]
        x = self.flatten(inputs)    # [batch_size, 784]
        x = self.dense1(x)          # [batch_size, 1000]
        x = self.dense2(x)          # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


if __name__ == "__main__":
    train_path = "/root/xiaoxuan/data/mnist/train.csv"
    test_path = "/root/xiaoxuan/data/mnist/test.csv"
    loader = MnistDataLoader(train_path, test_path)
    print("数据加载完成...")

    batch_size = 50
    learning_rate = 0.001

    model = MLP()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for step in range(2000):
        X, y = loader.get_batch(batch_size)

        with tf.GradientTape() as tape:
            y_pred = model(X)

            # categorical_crossentropy | sparse_categorical_crossentropy
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

        if step % 100 == 0:
            # 模型评估
            test_y_ = model(loader.test_x)
            sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            sparse_categorical_accuracy.update_state(y_true=loader.test_y, y_pred=test_y_)
            print("step: %d" % step, "test accuracy: %f" % sparse_categorical_accuracy.result(), "loss: %f" % loss)










