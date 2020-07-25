import os
import tensorflow as tf

"""
    FRecord 可以理解为一系列序列化的 tf.train.Example 元素所组成的列表文件，而每一个 tf.train.Example 又由若干个 tf.train.Feature 的字典组成

     步骤：
        读取该数据元素到内存；
        将该元素转换为 tf.train.Example 对象（每一个 tf.train.Example 由若干个 tf.train.Feature 的字典组成，因此需要先建立 Feature 的字典）；
        将该 tf.train.Example 对象序列化为字符串，并通过一个预先定义的 tf.io.TFRecordWriter 写入 TFRecord 文件。
     读取步骤：
        通过 tf.data.TFRecordDataset 读入原始的 TFRecord 文件（此时文件中的 tf.train.Example 对象尚未被反序列化），获得一个 tf.data.Dataset 数据集对象；
        通过 Dataset.map 方法，对该数据集对象中的每一个序列化的 tf.train.Example 字符串执行 tf.io.parse_single_example 函数，从而实现反序列化。
      
    数据格式：  
        tf.train.BytesList ：字符串或原始 Byte 文件（如图片），通过 bytes_list 参数传入一个由字符串数组初始化的 tf.train.BytesList 对象；
        tf.train.FloatList ：浮点数，通过 float_list 参数传入一个由浮点数数组初始化的 tf.train.FloatList 对象；
        tf.train.Int64List ：整数，通过 int64_list 参数传入一个由整数数组初始化的 tf.train.Int64List 对象。
        如果只希望保存一个元素而非数组，传入一个只有一个元素的数组即可。

    反序列化时需要指定元数据信息， 
        tf.io.FixedLenFeature 的三个输入参数 shape 、 dtype 和 default_value （可省略）为每个 Feature 的形状、类型和默认值。
        这里我们的数据项都是单个的数值或者字符串，所以 shape 为空数组。

    tf.train.example
        tf.train.Features   -> {"key": tf.train.Feature}
            tf.train.Feature    -> tf.train.ByteList, tf.train.FloatList, tf.train.Int64List
"""


favorite_books = [name.encode('utf-8') for name in ['machine learning', 'cc150']]
favorite_books_bytelist = tf.train.BytesList(value=favorite_books)
hours_floatlist = tf.train.FloatList(value=[15.5, 9.5, 70, 80])
age_int64list = tf.train.Int64List(value=[42])

features = tf.train.Features(
    feature={
        "favorite_books": tf.train.Feature(bytes_list=favorite_books_bytelist),
        "hours": tf.train.Feature(float_list=hours_floatlist),
        "age": tf.train.Feature(int64_list=age_int64list)
    }
)
print(features)

# 创建一个example对象
example = tf.train.Example(features=features)
print(example)

# 序列化
serialized_example = example.SerializeToString()
print(serialized_example)


# 持久化到磁盘
output_dir = 'tfrecord_basic'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
filename = "test.tfrecords"
filename_fullpath = os.path.join(output_dir, filename)
with tf.io.TFRecordWriter(filename_fullpath) as writer:
    for i in range(3):
        writer.write(serialized_example)


# 加载tfrecord文件
dataset = tf.data.TFRecordDataset([filename_fullpath])
for serialized_example_tensor in dataset:
    print(serialized_example_tensor)


# 将序列化的进行反序列化
expected_features = {
    "favorite_books": tf.io.VarLenFeature(dtype=tf.string),
    "hours": tf.io.VarLenFeature(dtype=tf.float32),
    "age": tf.io.FixedLenFeature([], dtype=tf.int64),
}
for serialized_example_tensor in dataset:
    example = tf.io.parse_single_example(serialized_example_tensor, expected_features)
    books = tf.sparse.to_dense(example["favorite_books"], default_value=b"")
    for book in books:
        print(book.numpy().decode("UTF-8"))


# 将tfrecord存储为压缩文件
filename_fullpath_zip = filename_fullpath + '.zip'
options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter(filename_fullpath_zip, options) as writer:
    for i in range(3):
        writer.write(serialized_example)


dataset_zip = tf.data.TFRecordDataset([filename_fullpath_zip], compression_type="GZIP")
for serialized_example_tensor in dataset_zip:
    example = tf.io.parse_single_example(
        serialized_example_tensor,
        expected_features)
    books = tf.sparse.to_dense(example["favorite_books"], default_value=b"")
    for book in books:
        print(book.numpy().decode("UTF-8"))





