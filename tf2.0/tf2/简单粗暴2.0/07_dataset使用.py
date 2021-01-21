import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

"""
    https://tf.wiki/zh/basic/tools.html
    
    dataSet 的使用
        tf.data.Dataset 由一系列的可迭代访问的元素（element）组成，每个元素包含一个或多个张量。比如说，对于一个由图像组成的数据集，
        每个元素可以是一个形状为 长×宽×通道数 的图片张量，也可以是由图片张量和图片标签张量组成的元组（Tuple）。
        
    tf.data.Dataset.from_tensor_slices() ，适用于数据量较小（能够整个装进内存）的情况
        数据集的元素数量为张量第 0 维的大小
        当提供多个张量作为输入时，张量的第 0 维大小必须相同，且必须将多个张量作为元组（Tuple，即使用 Python 中的小括号）拼接并作为输入。
        # X = np.array([2013, 2014, 2015, 2016, 2017])
        # Y = np.array([12000, 14000, 15000, 16500, 17500])
        dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        
    对于特别巨大而无法完整载入内存的数据集，我们可以先将数据集处理为 TFRecord 格式，然后使用 tf.data.TFRocrdDataset()
    
    
    tf.data.Dataset 类为我们提供了多种数据集预处理方法。最常用的如：
        Dataset.map(f) ：对数据集中的每个元素应用函数 f ，得到一个新的数据集（这部分往往结合 tf.io 进行读写和解码文件， tf.image 进行图像处理）；
            num_parallel_calls=2    此参数可以调节并行度
        Dataset.shuffle(buffer_size) ：将数据集打乱（设定一个固定大小的缓冲区（Buffer），取出前 buffer_size 个元素放入，
                并从缓冲区中随机采样，采样后的数据用后续数据替换）；tf.data.Dataset 作为一个针对大规模数据设计的迭代器，本身无法方便地获得自身元素的数量或随机访问元素
                因此，缓冲区的大小需要根据数据集的特性和数据排列顺序特点来进行合理的设置
        Dataset.batch(batch_size) ：将数据集分成批次，即对每 batch_size 个元素，使用 tf.stack() 在第 0 维合并，成为一个元素；
        Dataset.repeat() （重复数据集的元素）、 
        Dataset.reduce() （与 Map 相对的聚合操作）、 
        Dataset.take() （截取数据集中的前若干个元素）等
        
        Dataset.prefetch() 方法，使得我们可以让数据集对象 Dataset 在训练时预取出若干个元素， 使得在 GPU 训练的同时 CPU 可以准备数据，从而提升训练流程的效率
            mnist_dataset = mnist_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            此处参数 buffer_size 既可手工设置
    
    dataset可以使用for循环进行遍历，也可以使用迭代器，next(it)来使用 -> it = iter(dataset) ; next(it)
    Keras 支持使用 tf.data.Dataset 直接作为输入。当调用 tf.keras.Model 的 fit() 和 evaluate() 方法时，
    可以将参数中的输入数据 x 指定为一个元素格式为 (输入数据, 标签数据) 的 Dataset ，并忽略掉参数中的标签数据 y
        
"""

dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))     # TensorSliceDataset

# 可以用for循环遍历
for item in dataset:
    print(item)


# 将数据集重复三次，每次拿取7条数据
dataset = dataset.repeat(3).batch(7)        # BatchDataset
for item in dataset:
    print(item)
# tf.Tensor([0 1 2 3 4 5 6], shape=(7,), dtype=int64)
# tf.Tensor([7 8 9 0 1 2 3], shape=(7,), dtype=int64)
# tf.Tensor([4 5 6 7 8 9 0], shape=(7,), dtype=int64)
# tf.Tensor([1 2 3 4 5 6 7], shape=(7,), dtype=int64)
# tf.Tensor([8 9], shape=(2,), dtype=int64)


# 对BatchDataset 中每个元素进行处理，每个元素就相当于是一个batch
# 和map的区别相当于是将多个dataset合成了一个，map是一对一
dataset2 = dataset.interleave(
    lambda v: tf.data.Dataset.from_tensor_slices(v),
    cycle_length=5,     # 一次同时处理dataset多少元素
    block_length=5,     # 每次取多少个结果出来
)
for item in dataset2:
    print(item)
# tf.Tensor(0, shape=(), dtype=int64)
# tf.Tensor(1, shape=(), dtype=int64)
# tf.Tensor(2, shape=(), dtype=int64)
# tf.Tensor(3, shape=(), dtype=int64)
# tf.Tensor(4, shape=(), dtype=int64)
# tf.Tensor(7, shape=(), dtype=int64)
# tf.Tensor(8, shape=(), dtype=int64)
# tf.Tensor(9, shape=(), dtype=int64)
# tf.Tensor(0, shape=(), dtype=int64)
# tf.Tensor(1, shape=(), dtype=int64)
# tf.Tensor(4, shape=(), dtype=int64)
# tf.Tensor(5, shape=(), dtype=int64)
# tf.Tensor(6, shape=(), dtype=int64)
# tf.Tensor(7, shape=(), dtype=int64)
# tf.Tensor(8, shape=(), dtype=int64)
# tf.Tensor(1, shape=(), dtype=int64)
# tf.Tensor(2, shape=(), dtype=int64)
# tf.Tensor(3, shape=(), dtype=int64)
# tf.Tensor(4, shape=(), dtype=int64)
# tf.Tensor(5, shape=(), dtype=int64)
# tf.Tensor(8, shape=(), dtype=int64)
# tf.Tensor(9, shape=(), dtype=int64)
# tf.Tensor(5, shape=(), dtype=int64)
# tf.Tensor(6, shape=(), dtype=int64)
# tf.Tensor(2, shape=(), dtype=int64)
# tf.Tensor(3, shape=(), dtype=int64)
# tf.Tensor(9, shape=(), dtype=int64)
# tf.Tensor(0, shape=(), dtype=int64)
# tf.Tensor(6, shape=(), dtype=int64)
# tf.Tensor(7, shape=(), dtype=int64)


# 字典和矩阵
x = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array(['cat', 'dog', 'fox'])
dataset3 = tf.data.Dataset.from_tensor_slices((x, y))
for item_x, item_y in dataset3:
    print(item_x, item_y)
    print(item_x.numpy(), item_y.numpy())
# tf.Tensor([1 2], shape=(2,), dtype=int64) tf.Tensor(b'cat', shape=(), dtype=string)
# [1 2] b'cat'
# tf.Tensor([3 4], shape=(2,), dtype=int64) tf.Tensor(b'dog', shape=(), dtype=string)
# [3 4] b'dog'
# tf.Tensor([5 6], shape=(2,), dtype=int64) tf.Tensor(b'fox', shape=(), dtype=string)
# [5 6] b'fox'


dataset4 = tf.data.Dataset.from_tensor_slices({"feature": x, "label": y})
for item in dataset4:
    print(item)




"""
    操作CSV
    将一个目录下的所有的csv
        1、filename -> dataset
        2、read file -> dataset -> datasets -> merge
        3、parse csv
"""


path = r"/tf2.0/tf2/note/generate_csv"
train_files = []
for file in os.listdir(path):
    if "train" in file:
        train_files.append(os.path.join(path, file))

# 传入一个文件名列表
filename_dataset = tf.data.Dataset.list_files(train_files)
for filename in filename_dataset:
    print(filename)


n_readers = 5
# 和map的区别相当于是将多个dataset合成了一个，map是一对一
dataset = filename_dataset.interleave(
    lambda filename: tf.data.TextLineDataset(filename).skip(1),  # skip(1) 表示跳过第一行
    cycle_length=n_readers
)
for line in dataset.take(15):
    print(line.numpy())


# tf.io.decode_csv(str,record_defaults) 解析csv
sample_str = '1,2,3,4,5'
record_defaults = [tf.constant(0, dtype=tf.int32), 0, np.nan, "hello", tf.constant([])]     # 指明元素类型以及缺失时的默认值
parsed_fields = tf.io.decode_csv(sample_str, record_defaults)
print(parsed_fields)


# 会抛异常的两种情况
try:
    parsed_fields = tf.io.decode_csv(',,,,', record_defaults)
except tf.errors.InvalidArgumentError as ex:
    print(ex)

try:
    parsed_fields = tf.io.decode_csv('1,2,3,4,5,6,7', record_defaults)
except tf.errors.InvalidArgumentError as ex:
    print(ex)



"""
    处理房价预测csv数据
"""
def parse_csv_line(line, n_fields):
    defs = [tf.constant(np.nan)] * n_fields
    parsed_fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(parsed_fields[0:-1])  # tensor列表转变为tensor向量
    y = tf.stack(parsed_fields[-1:])
    return x, y


# 1、filename -> dataset
# 2、read file -> dataset -> dataset -> merge
# 3、parse csv
def csv_reader_dataset(filenames, n_readers=5, batch_size=32, n_parse_threads=5, shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()      # 没有加参数就是重复无限次
    # 和map的区别相当于是将多个dataset合成了一个，map是一对一
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename).skip(1),
        cycle_length=n_readers
    )
    # buffer_size 相当于是缓冲区大小
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_csv_line, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset


train_set = csv_reader_dataset(train_files, batch_size=3)
batch_size = 32
train_set = csv_reader_dataset(train_filename, batch_size=batch_size)
valid_set = csv_reader_dataset(valid_filename, batch_size=batch_size)
test_set = csv_reader_dataset(test_filename, batch_size=batch_size)


model = tf.keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=[8]),
    keras.layers.Dense(1)
])
model.compile(loss="mean_squared_error", optimizer='sgd')
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
history = model.fit(train_set, validation_data=valid_set,
                    steps_per_epoch=11160 // batch_size,         # 因为是自己定义的数据集，需要告诉tf多少步算一个epoch
                    validation_steps=3870 // batch_size,        # 验证集需要迭代的次数
                    epochs=100, callbacks=callbacks)
model.evaluate(test_set, steps=5160 // batch_size)

























