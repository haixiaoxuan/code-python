from __future__ import absolute_import, division, print_function, unicode_literals


# 单机从HDFS拉取 TFRecord 格式的数据进行训练


def main_fun(path, buffer_size, batch_size, epochs):
    """
        从 HDFS 读取 TFRecord 格式数据作为训练数据
    """
    import tensorflow as tf
    print(tf.__version__)
    BUFFER_SIZE = buffer_size
    BATCH_SIZE = batch_size

    def parse_tfos(example_proto):
        """
            解析 HDFS 上的 TFRecord格式的数据
        """
        expected_features = {
            "features": tf.io.FixedLenFeature(784, dtype=tf.float32),
            "label": tf.io.FixedLenFeature([], dtype=tf.int64),
        }
        example = tf.io.parse_single_example(example_proto, expected_features)
        image = tf.reshape(example["features"], (28, 28, 1))
        label = example["label"]
        return (image, label)


    # 从 HDFS 读取 tfrecord文件并作处理
    print(path)
    ds = tf.data.Dataset.list_files(path)
    ds = ds.repeat(epochs).shuffle(BUFFER_SIZE)
    ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=-1)
    train_datasets_unbatched = ds.map(parse_tfos)


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

    # single node
    single_worker_model = build_and_compile_cnn_model()
    train_datasets = train_datasets_unbatched.batch(BATCH_SIZE)
    single_worker_model.fit(x=train_datasets, epochs=3)


if __name__ == "__main__":

    from pyspark.context import SparkContext
    from pyspark.conf import SparkConf
    sc = SparkContext(conf=SparkConf().setAppName("mnist_keras"))

    import os
    os.environ.setdefault("LIB_JVM", "/usr/lib/java/jdk1.8.0_144/jre/lib/amd64/server")
    os.environ.setdefault("LIB_HDFS", "/opt/cloudera/parcels/CDH-5.13.3-1.cdh5.13.3.p0.2/lib64")

    path = r"hdfs://master01.hadoop.dtmobile.cn:8020/home/mnist-test/data/mnist.tfrecord"
    buffer_size = 10000
    batch_size = 128
    epochs = 3
    main_fun(path, buffer_size, batch_size, epochs)

    sc.stop()

