import tensorflow as tf
from MLP_mnist import MnistDataLoader


class RNN(tf.keras.Model):
    def __init__(self, data_length, batch_size, seq_length):
        super().__init__()
        self.data_length = data_length
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.cell = tf.keras.layers.LSTMCell(units=256)
        self.dense = tf.keras.layers.Dense(units=10)    # 十分类

    def call(self, inputs):
        inputs = tf.reshape(inputs, (-1, self.seq_length, self.data_length))       # [batch_size, seq_length, data_length]
        state = self.cell.get_initial_state(batch_size=inputs.shape[0], dtype=tf.float32)
        for t in range(self.seq_length):
            output, state = self.cell(inputs[:, t, :], state)
        logits = self.dense(output)
        return tf.nn.softmax(logits)


if __name__ == "__main__":
    train_path = "/root/xiaoxuan/data/mnist/train.csv"
    test_path = "/root/xiaoxuan/data/mnist/test.csv"
    loader = MnistDataLoader(train_path, test_path)
    print("数据加载完成...")

    data_length = 28
    seq_length = 28
    batch_size = 50
    learning_rate = 1e-3

    model = RNN(data_length=data_length, batch_size=batch_size, seq_length=seq_length)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for step in range(1000):
        X, y = loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

        if step % 100 == 0:
            test_y_ = model(loader.test_x)
            sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            sparse_categorical_accuracy.update_state(y_true=loader.test_y, y_pred=test_y_)
            print("step: %d" % step, "test accuracy: %f" % sparse_categorical_accuracy.result(), "loss: %f" % loss)






















