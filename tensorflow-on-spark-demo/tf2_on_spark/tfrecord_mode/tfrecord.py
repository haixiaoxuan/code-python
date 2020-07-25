from __future__ import absolute_import, division, print_function, unicode_literals


def main_fun(args, ctx):
    import tensorflow as tf
    from tensorflowonspark import compat

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    BUFFER_SIZE = args.buffer_size
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.cluster_size


    def parse_tfos(example_proto):
        expected_features = {
            "features": tf.io.FixedLenFeature(784, dtype=tf.float32),
            "label": tf.io.FixedLenFeature([], dtype=tf.int64),
        }
        example = tf.io.parse_single_example(example_proto, expected_features)

        image = tf.reshape(example["features"], (28, 28, 1))
        label = example["label"]
        return (image, label)

    image_pattern = ctx.absolute_path(args.images_labels)   # 获取HDFS绝对路径
    print("===============> 绝对路径: " + image_pattern)


    ds = tf.data.Dataset.list_files(image_pattern)
    ds = ds.repeat().shuffle(BUFFER_SIZE)
    ds = ds.interleave(tf.data.TFRecordDataset)
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
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            metrics=['accuracy'])
        return model


    GLOBAL_BATCH_SIZE = BATCH_SIZE * NUM_WORKERS
    train_datasets = train_datasets_unbatched.batch(GLOBAL_BATCH_SIZE)

    # this fails
    # callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=args.model_dir)]
    tf.io.gfile.makedirs(args.model_dir)
    filepath = args.model_dir + "/weights-{epoch:04d}"
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=1, save_weights_only=True)]


    steps_per_epoch = 55000 / GLOBAL_BATCH_SIZE

    with strategy.scope():
        multi_worker_model = build_and_compile_cnn_model()
    multi_worker_model.fit(x=train_datasets,
                           epochs=args.epochs,
                           steps_per_epoch=steps_per_epoch,
                           callbacks=callbacks)


    # 不注释会执行失败，猜测可能原因是不支持多个task往一个目录里面保存模型
    # from tensorflow_estimator.python.estimator.export import export_lib
    # export_dir = export_lib.get_timestamped_export_dir(args.export_dir)
    # compat.export_saved_model(multi_worker_model, export_dir, ctx.job_name == 'chief')


if __name__ == '__main__':
    import argparse
    from pyspark.context import SparkContext
    from pyspark.conf import SparkConf
    from tensorflowonspark import TFCluster

    sc = SparkContext(conf=SparkConf().setAppName("mnist_keras"))
    executors = sc._conf.get("spark.executor.instances")
    num_executors = int(executors) if executors is not None else 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="number of records per batch", type=int, default=64)
    parser.add_argument("--buffer_size", help="size of shuffle buffer", type=int, default=10000)
    parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
    parser.add_argument("--data_format", help="data format (tfos|tfds)", type=str, choices=["tfos", "tfds"],
                        default="tfos")
    parser.add_argument("--epochs", help="number of epochs", type=int, default=30)
    parser.add_argument("--images_labels", help="HDFS path to MNIST image_label files in parallelized format")
    parser.add_argument("--model_dir", help="path to save model/checkpoint", default="mnist_model")
    parser.add_argument("--export_dir", help="path to export saved_model", default="mnist_export")
    parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")

    args = parser.parse_args()
    print("args:", args)

    cluster = TFCluster.run(sc, main_fun, args, args.cluster_size, num_ps=0, tensorboard=args.tensorboard,
                            input_mode=TFCluster.InputMode.TENSORFLOW, master_node='chief')
    cluster.shutdown()


