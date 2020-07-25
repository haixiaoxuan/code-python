import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from tensorflow import keras

start = time.time()

def load_mnist():
    # 离线数据存放路径
    data_dir = r"C:\Users\xiexiaoxuan\PycharmProjects\ml_project\tf2\notebook\data\\"
    import gzip

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    trX = extract_data(data_dir + 'train-images-idx3-ubyte.gz', 60000, 16, 28 * 28).reshape((60000, 28, 28))
    trY = extract_data(data_dir + 'train-labels-idx1-ubyte.gz', 60000, 8, 1).reshape((60000))
    teX = extract_data(data_dir + 't10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28).reshape((10000, 28, 28))
    teY = extract_data(data_dir + 't10k-labels-idx1-ubyte.gz', 10000, 8, 1).reshape((10000))

    return (trX, trY), (teX, teY)


(x_train_all, y_train_all), (x_test, y_test) = load_mnist()
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000].astype(np.int), y_train_all[5000:].astype(np.int)
print(x_valid.shape, x_train.shape)


# 对数据进行归一化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1)
x_valid_scaled = scaler.transform(x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1)
x_test_scaled = scaler.transform(x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1)


# 构造卷积神经网络模型
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


# 如果y已经经过one-hot编码，则直接使用 categorical_crossentropy 损失函数
# metrics 表示要统计什么指标
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
# history = model.fit(x_train_scaled, y_train, epochs=1, validation_data=(x_valid_scaled, y_valid))
history = model.fit(x_train_scaled, y_train, epochs=1)


def draw_res(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 3)
    plt.show()

draw_res(history)

print("耗时", str(time.time()-start))
