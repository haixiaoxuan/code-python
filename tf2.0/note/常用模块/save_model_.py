import tensorflow as tf


"""
    SavedModel 包含了一个 TensorFlow 程序的完整信息： 不仅包含参数的权值，还包含计算的流程（即计算图）
    当模型导出为 SavedModel 文件时，无需建立模型的源代码即可再次运行模型
"""

# 因为 SavedModel 基于计算图，所以对于使用继承 tf.keras.Model 类建立的 Keras 模型，
# 其需要导出到 SavedModel 格式的方法（比如 call ）都需要使用 @tf.function 修饰
# 对于使用继承 tf.keras.Model 类建立的 Keras 模型 model ，使用 SavedModel 载入后将无法使用 model() 直接进行推断，而需要使用 model.call() 。
tf.saved_model.save(model, "保存的目标文件夹名称")

# 加载
model = tf.saved_model.load("保存的目标文件夹名称")























