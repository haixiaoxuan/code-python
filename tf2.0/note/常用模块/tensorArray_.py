import tensorflow as tf

"""
    在部分网络结构，尤其是涉及到时间序列的结构中，我们可能需要将一系列张量以数组的方式依次存放起来，
    以供进一步处理。当然，在 Eager Execution 下，你可以直接使用一个 Python 列表（List）存放数组。
    不过，如果你需要基于计算图的特性（例如使用 @tf.function 加速模型运行或者使用 SavedModel 导出模型），
    就无法使用这种方式了。因此，TensorFlow 提供了 tf.TensorArray ，一种支持计算图特性的 TensorFlow 动态数组。
"""


@tf.function
def array_write_and_read():
    arr = tf.TensorArray(dtype=tf.float32, size=3)
    arr = arr.write(0, tf.constant(0.0))
    arr = arr.write(1, tf.constant(1.0))
    arr = arr.write(2, tf.constant(2.0))
    arr_0 = arr.read(0)
    arr_1 = arr.read(1)
    arr_2 = arr.read(2)
    return arr_0, arr_1, arr_2

a, b, c = array_write_and_read()
print(a, b, c)