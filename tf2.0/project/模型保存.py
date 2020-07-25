import tensorflow as tf
from CNN_mnist import CNN
from MLP_mnist import MnistDataLoader


def train(loader):
    batch_size = 50
    learning_rate = 0.001

    model = CNN()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    checkpoint = tf.train.Checkpoint(myModel=model)

    for step in range(2000):
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
            checkpoint.save('./save/model.ckpt')


def test(loader):
    model_to_be_restored = CNN()  # 待恢复参数的同一模型
    checkpoint = tf.train.Checkpoint(myModel=model_to_be_restored)
    checkpoint.restore("./save/model.ckpt-20")

    test_y_ = model_to_be_restored(loader.test_x.reshape(-1, 28, 28, 1))
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    sparse_categorical_accuracy.update_state(y_true=loader.test_y, y_pred=test_y_)
    print("test accuracy: %f" % sparse_categorical_accuracy.result())


if __name__ == "__main__":
    train_path = "/root/xiaoxuan/data/mnist/train.csv"
    test_path = "/root/xiaoxuan/data/mnist/test.csv"
    loader = MnistDataLoader(train_path, test_path)
    print("数据加载完成...")

    # train(loader)
    test(loader)
