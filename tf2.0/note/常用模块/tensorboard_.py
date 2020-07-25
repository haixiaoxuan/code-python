import tensorflow as tf

"""
    使用：
        打开终端输入：tensorboard --logdir=./tensorboard --bind_all
        然后访问 6006 端口
    note：
        如果需要重新训练，需要删除掉记录文件夹内的信息并重启 TensorBoard（或者建立一个新的记录文件夹并开启 TensorBoard， --logdir 参数设置为新建立的文件夹）；
        记录文件夹目录保持全英文。
"""

# 1. 首先在代码目录下建立一个文件夹（如 ./tensorboard ）存放 TensorBoard 的记录文件，并在代码中实例化一个记录器
summary_writer = tf.summary.create_file_writer('./tensorboard')


# 接下来，当需要记录训练过程中的参数时，通过 with 语句指定希望使用的记录器，
# 并对需要记录的参数（一般是 scalar）运行 tf.summary.scalar(name, tensor, step=batch_index) ，
# 即可将训练过程中参数在 step 时候的值记录下来。这里的 step 参数可根据自己的需要自行制定，一般可设置为当前训练过程中的 batch 序号。

summary_writer = tf.summary.create_file_writer('./tensorboard')
# 开始模型训练
for batch_index in range(num_batches):
    # ...（训练代码，当前batch的损失值放入变量loss中）
    with summary_writer.as_default():                               # 希望使用的记录器
        tf.summary.scalar("loss", loss, step=batch_index)
        tf.summary.scalar("MyScalar", my_scalar, step=batch_index)  # 还可以添加其他自定义的变量





































