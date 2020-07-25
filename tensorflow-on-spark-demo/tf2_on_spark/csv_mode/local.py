"""
    单机测试，性能对比

    当epoch增多时，分布式快于单机，但是分布式准确率提升很慢，可能是参数的问题
"""

import numpy as np
import tensorflow as tf
import time


path = r"C:\Users\xiaoxuan\Desktop\work\tensorflow\data\mnist.csv"
batch_size = 64
epoch = 200

start = time.time()
def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    return model


def rdd_generator():
    while True:
        with open(path, "r") as f:
            for line in f:
                arr = [float(i) for i in line.split(",")]
                image = np.array(arr[:-1]).astype(np.float32)
                image = np.reshape(image, (28, 28, 1))
                label = np.array(arr[-1]).astype(np.int)
                label = np.reshape(label, (1,))
                yield (image, label)


ds = tf.data.Dataset.from_generator(rdd_generator, (tf.float32, tf.float32),
                                    (tf.TensorShape([28, 28, 1]), tf.TensorShape([1])))
ds = ds.batch(batch_size)


multi_worker_model = build_and_compile_cnn_model()
steps_per_epoch = 50000 / batch_size

multi_worker_model.fit(x=ds, epochs=epoch, steps_per_epoch=steps_per_epoch)
end = time.time()

print("耗时:", end-start)



