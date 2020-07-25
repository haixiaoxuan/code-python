import tensorflow as tf
import numpy as np

"""
    tf.data 的核心是 tf.data.Dataset 类，提供了对数据集的高层封装
    tf.data.Dataset 由一系列的可迭代访问的元素（element）组成，每个元素包含一个或多个张量。比如说，对于一个由图像组成的数据集，每个元素可以是一个形状为 长×宽×通道数 的图片张量，也可以是由图片张量和图片标签张量组成的元组（Tuple）。
    使用Dataset之后，可以对dataset直接训练 ，(输入数据, 标签数据)
        model.fit(mnist_dataset, epochs=num_epochs)
"""


# 最基础的建立 tf.data.Dataset 的方法是使用 tf.data.Dataset.from_tensor_slices() ，
# 适用于数据量较小（能够整个装进内存）的情况。具体而言，如果我们的数据集中的所有元素通过张量的第 0 维，
# 拼接成一个大的张量（例如，前节的 MNIST 数据集的训练集即为一个 [60000, 28, 28, 1] 的张量，
# 表示了 60000 张 28*28 的单通道灰度图像），那么我们提供一个这样的张量或者第 0 维大小相同的多个张量作为输入，
# 即可按张量的第 0 维展开来构建数据集，数据集的元素数量为张量第 0 位的大小。具体示例如下：

X = tf.constant([2013, 2014, 2015, 2016, 2017])
Y = tf.constant([12000, 14000, 15000, 16500, 17500])

# 也可以使用NumPy数组，效果相同
# X = np.array([2013, 2014, 2015, 2016, 2017])
# Y = np.array([12000, 14000, 15000, 16500, 17500])

dataset = tf.data.Dataset.from_tensor_slices((X, Y))

for x, y in dataset:
    print(x.numpy(), y.numpy())


"""
    数据预处理方法
"""

# Dataset.map(f) ：对数据集中的每个元素应用函数 f ，得到一个新的数据集（这部分往往结合 tf.io 进行读写和解码文件， tf.image 进行图像处理）；
# Dataset.shuffle(buffer_size) ：将数据集打乱（设定一个固定大小的缓冲区（Buffer），取出前 buffer_size 个元素放入，并从缓冲区中随机采样，采样后的数据用后续数据替换）；
# Dataset.batch(batch_size) ：将数据集分成批次，即对每 batch_size 个元素，使用 tf.stack() 在第 0 维合并，成为一个元素。
# Dataset.prefetch() ：预取出数据集中的若干个元素


# 例
def rot90(image, label):
    image = tf.image.rot90(image)
    return image, label

mnist_dataset = mnist_dataset.map(rot90)






