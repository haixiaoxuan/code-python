import tensorflow as tf

"""
    tf.config：GPU 的使用与分配
    默认情况下 TensorFlow 会使用其所能够使用的所有 GPU
"""


# 获得当前主机上某种特定运算设备类型（如 GPU 或 CPU ）的列表
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)


# 设置当前程序可见的设备范围
# 限定当前程序只使用下标为 0、1 的两块显卡（GPU:0 和 GPU:1）
# 使用环境变量 CUDA_VISIBLE_DEVICES 也可以控制程序所使用的 GPU。假设发现四卡的机器上显卡 0,1 使用中，显卡 2,3 空闲，Linux 终端输入:
# export CUDA_VISIBLE_DEVICES=2,3
# 或者在代码中输入 os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
tf.config.experimental.set_visible_devices(devices=gpus[0:2], device_type='GPU')


"""
    默认情况下，TensorFlow 将使用几乎所有可用的显存，以避免内存碎片化所带来的性能损失。不过，TensorFlow 提供两种显存使用策略，让我们能够更灵活地控制程序的显存使用方式：
        1. 仅在需要时申请显存空间（程序初始运行时消耗很少的显存，随着程序的运行而动态申请显存）；
        2. 限制消耗固定大小的显存（程序不会超出限定的显存大小，若超出的报错）。
"""


# 1
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, True)


# 2 设置 TensorFlow 固定消耗 GPU:0 的 1GB 显存
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])


"""
      单 GPU 模拟多 GPU 环境   
"""
# 在实体 GPU GPU:0 的基础上建立了两个显存均为 2GB 的虚拟 GPU。
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048),
     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])








