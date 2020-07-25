import tensorflow as tf
import os


# 设置日志输出
tf.debugging.set_log_device_placement(True)


# 获取所有物理GPU
gpus = tf.config.experimental.list_physical_devices("GPU")


# 如果想指定使用某块GPU,第二个参数表示设备类型
# 也可以通过环境变量的方式来指定
tf.config.experimental.set_visible_devices(gpus[3], "GPU")
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"


# 设置使用GPU随内存情况而定，不独占
tf.config.experimental.set_memory_growth(gpus[0], True)


# 设置GPU逻辑切分
tf.config.experimental.set_virtual_device_configuration(gpus[1],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072),
                                                         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])


# 列出所有逻辑GPU
logical_gpus = tf.config.experimental.list_logical_devices("GPU")
print("GPU ", gpus)


# 手动指定GPU设备, 当模型特别大的时候，可以把不同的层放到不同的GPU上
for logical_gpu in logical_gpus:
    with tf.device(logical_gpu.name):
        """ 计算逻辑 """
with tf.device("/CPU:0"):
    """ 计算逻辑 """

# 自动调整
tf.config.set_soft_device_placement(True)

