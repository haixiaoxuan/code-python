from tensorflow import keras
import tensorflow as tf

"""
    https://tf.wiki/zh/basic/tools.html

    Checkpoint 只保存模型的参数，不保存模型的计算过程，因此一般用于在具有模型源代码的时候恢复之前训练好的模型参数。
    如果需要导出模型（无需源代码也能运行模型），请使用 SavedModel 。
    
     TensorFlow 提供了 tf.train.Checkpoint 这一强大的变量保存与恢复类，可以使用其 save() 和 restore() 方法将 TensorFlow 中所有包含 
     Checkpointable State 的对象进行保存和恢复。具体而言，tf.keras.optimizer 、 tf.Variable 、 tf.keras.Layer 或者 tf.keras.Model 实例都可以被保存。
     其使用方法非常简单，我们首先声明一个 Checkpoint
     
     HDF5 格式, 就是 keras 训练后的模型，其中已经包含了训练后的模型结构和权重等信息。
    
"""


# 这里 tf.train.Checkpoint() 接受的初始化参数比较特殊，是一个 **kwargs 。具体而言，是一系列的键值对，键名可以随意取，值为需要保存的对象, 注意，在恢复变量的时候，我们还将使用这一键名
checkpoint = tf.train.Checkpoint(model=model)
checkpoint = tf.train.Checkpoint(myAwesomeModel=model, myAwesomeOptimizer=optimizer)

# 接下来，当模型训练完成需要保存的时候，使用：
checkpoint.save(save_path_with_prefix)

# 恢复
model_to_be_restored = MyModel()                                        # 待恢复参数的同一模型
checkpoint = tf.train.Checkpoint(myAwesomeModel=model_to_be_restored)   # 键名保持为“myAwesomeModel”
checkpoint.restore(save_path_with_prefix_and_index)     # tf.train.latest_checkpoint(save_path) 可以得到最后一个 index



"""
    tf.train.CheckpointManager
        在长时间的训练后，程序会保存大量的 Checkpoint，但我们只想保留最后的几个 Checkpoint；
        Checkpoint 默认从 1 开始编号，每次累加 1，但我们可能希望使用别的编号方式（例如使用当前 Batch 的编号作为文件编号）
"""
checkpoint = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(checkpoint, directory='./save', checkpoint_name='model.ckpt', max_to_keep=k)
manager.save()
manager.save(checkpoint_number=100)



"""
    HDF5 保存模型，在callback中使用
"""
output_model_file = "xx.h5"
keras.callbacks.ModelCheckpoint(output_model_file,
                                save_best_only=True,
                                save_weights_only=False)


"""
    使用保存的模型
"""
model = keras.models.load_model(output_model_file)


# 如果只保存了模型参数，而没有保存模型结构，需要提前把模型定义好
model = keras.models.Sequential()
# 省略定义
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd")

model.load_weights(output_model_file)


# 直接使用model保存权重
model.save_weights(output_model_file)






