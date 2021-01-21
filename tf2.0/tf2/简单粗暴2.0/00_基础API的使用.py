import tensorflow as tf
import numpy as np


# 常量
constant = tf.constant([[1, 2], [3, 4]])
print(constant[0])
print(constant[:, 1])
print(constant[..., 1])
print(tf.constant(np.array([1, 2])))


# string
constant_string = tf.constant("abc")
print(constant_string)
print(tf.strings.length(constant_string))
print(tf.strings.length(constant_string, unit="UTF8_CHAR"))
print(tf.strings.unicode_decode(constant_string, "UTF8"))       # 将字符串转变为byte数组

constant_string = tf.constant(["cafe", "hh", "你好"])
print(tf.strings.unicode_decode(constant_string, "UTF8"))       # 不是规则矩阵，类型为 RaggedTensor
tf.ragged.constant([[1, 2], [3, 4, 5]])       # 不规则矩阵的书写方式, 使用to_tensor方法可以对缺失值进行填充


# sparse tensor
sparse_tensor = tf.SparseTensor(indices=[[0, 1], [0, 2]], values=[4, 5], dense_shape=[2, 3])    # 表示第0行的1，2 值分别为4 5
print(sparse_tensor)
print(tf.sparse.to_dense(sparse_tensor))    # 如果要to_dense indices 必须是拍好序的，否则需要手动排序 tf.sparse.reorder(sparse_tensor)
print(tf.sparse.sparse_dense_matmul(sparse_tensor, tf.constant([[1, 2], [3, 4], [5, 6]])))  # 稀疏 * 密集


# op
print(constant + 10)
print(tf.square(constant))
print(constant @ tf.transpose(constant))        # 矩阵相乘


# variable
variable = tf.Variable([1, 3])
print(variable)
print(variable.value())
print(variable.numpy())
variable.assign(variable * 2)       # 不能使用等号赋值
variable[0].assign(0)


tf.expand_dims(input_eval, 0)       # 扩展维度
tf.squeeze(predictions, 0)          # 缩减维度








