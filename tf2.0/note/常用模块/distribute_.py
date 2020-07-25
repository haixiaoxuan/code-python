import tensorflow as tf


"""
    tf.distribute.Strategy 提供了若干种分布式策略
"""

# 单机多卡
"""
    训练开始前，该策略在所有 N 个计算设备上均各复制一份完整的模型；
    每次训练传入一个批次的数据时，将数据分成 N 份，分别传入 N 个计算设备（即数据并行）；
    N 个计算设备使用本地变量（镜像变量）分别计算自己所获得的部分数据的梯度；
    使用分布式计算的 All-reduce 操作，在计算设备间高效交换梯度数据并进行求和，使得最终每个设备都有了所有设备的梯度之和；
    使用梯度求和的结果更新本地变量（镜像变量）；
    当所有设备均更新本地变量后，进行下一轮训练（即该并行策略是同步的）。
    默认情况下，TensorFlow 中的 MirroredStrategy 策略使用 NVIDIA NCCL 进行 All-reduce 操作。
"""


# tf.distribute.MirroredStrategy 是一种简单且高性能的，数据并行的同步式分布式策略，主要支持多个 GPU 在同一台主机上训练。
# 使用这种策略时，我们只需实例化一个 MirroredStrategy 策略
# 可以在参数中指定设备，如: strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
strategy = tf.distribute.MirroredStrategy()

print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

# 并将模型构建的代码放入 strategy.scope() 的上下文环境中:
with strategy.scope():
    # 模型构建代码
    pass


# 多机训练
# MultiWorkerMirroredStrategy
# 参考 https://tf.wiki/zh/appendix/distributed.html





























