import tensorflow_datasets as tfds
import tensorflow as tf
import os
import json

tfds.disable_progress_bar()

BUFFER_SIZE = 10000
BATCH_SIZE = 64

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'task': {'type': 'worker', 'index': 1}
})

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()


# 将 MNIST 数据从 (0, 255] 缩放到 (0., 1.]
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
train_datasets_unbatched = datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)
train_datasets = train_datasets_unbatched.batch(BATCH_SIZE)


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


NUM_WORKERS = 2
# 由于`tf.data.Dataset.batch`需要全局的批处理大小，
# 因此此处的批处理大小按工作器数量增加。
# 以前我们使用64，现在变成128。
GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS
train_datasets = train_datasets_unbatched.batch(GLOBAL_BATCH_SIZE)


options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
train_datasets_no_auto_shard = train_datasets.with_options(options).repeat()


with strategy.scope():
    multi_worker_model = build_and_compile_cnn_model()
multi_worker_model.fit(x=train_datasets_no_auto_shard, epochs=3, steps_per_epoch=60000/GLOBAL_BATCH_SIZE)



