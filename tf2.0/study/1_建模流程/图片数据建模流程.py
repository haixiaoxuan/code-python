import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from matplotlib import pyplot as plt
import datetime
import pandas as pd

""" 处理图片数据可以使用两种方式：
        1. 使⽤tf.keras中的ImageDataGenerator⼯具构建图⽚数据⽣成器。
        2. 使⽤tf.data.Dataset搭配tf.image中的⼀些图⽚处理⽅法构建数据管道
    此处使用第二种方式
"""

BATCH_SIZE = 100


def load_image(img_path, size=(32, 32)):
    print(img_path)
    label = tf.constant(1, tf.int8) if tf.strings.regex_full_match(img_path, ".*/automobile/.*") else tf.constant(0, tf.int8)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)  # 注意此处为jpeg格式
    img = tf.image.resize(img, size) / 255.0
    return (img, label)


# 使⽤并⾏化预处理 num_parallel_calls 和预存数据prefetch来提升性能
ds_train = tf.data.Dataset.list_files("../../data/cifar2/train/*/*.jpg") \
    .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE)
ds_test = tf.data.Dataset.list_files("../../data/cifar2/test/*/*.jpg") \
    .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE)


# 查看部分样本
plt.figure(figsize=(8, 8))
for i, (img, label) in enumerate(ds_test.unbatch().take(9)):
    ax = plt.subplot(3, 3, i + 1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d" % label)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

for x,y in ds_train.take(1):
    print(x.shape, y.shape)      # x_shape(100, 32, 32, 3)


tf.keras.backend.clear_session() #清空会话
inputs = layers.Input(shape=(32,32,3))
x = layers.Conv2D(32,kernel_size=(3,3))(inputs)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(64,kernel_size=(5,5))(x)
x = layers.MaxPool2D()(x)
x = layers.Dropout(rate=0.1)(x)
x = layers.Flatten()(x)
x = layers.Dense(32,activation='relu')(x)
outputs = layers.Dense(1,activation = 'sigmoid')(x)
model = models.Model(inputs = inputs,outputs = outputs)

model.summary()
logdir = "./data/keras_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.binary_crossentropy,
    metrics=["accuracy"]
 )
history = model.fit(ds_train, epochs= 10, validation_data=ds_test,
                    callbacks = [tensorboard_callback], workers = 4)


dfhistory = pd.DataFrame(history.history)
dfhistory.index = range(1, len(dfhistory) + 1)
dfhistory.index.name = 'epoch'
dfhistory
# epoch  loss  accuracy  val_loss  val_accuracy




