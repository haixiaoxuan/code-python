import tensorflow as tf

"""
    tf.train.Checkpoint ：变量的保存与恢复
            Checkpoint 只保存模型的参数，不保存模型的计算过程，因此一般用于在具有模型源代码的时候恢复之前训练好的模型参数。
            如果需要导出模型（无需源代码也能运行模型），需要使用SavedModel 。
"""


# 接受的初始化参数比较特殊，是一个 **kwargs 。具体而言，是一系列的键值对，键名可以随意取，值为需要保存的对象
checkpoint = tf.train.Checkpoint(myModel=model)


# 当模型训练完成需要保存的时候
# 例如，在源代码目录建立一个名为 save 的文件夹并调用一次 checkpoint.save('./save/model.ckpt') ，
# 我们就可以在可以在 save 目录下发现名为 checkpoint 、 model.ckpt-1.index 、 model.ckpt-1.data-00000-of-00001 的三个文件，
# 这些文件就记录了变量信息。checkpoint.save() 方法可以运行多次，
# 每运行一次都会得到一个.index 文件和.data 文件，序号依次累加。
checkpoint.save("save_path_with_prefix")


# 在其他地方需要为模型重新载入之前保存的参数时
model_to_be_restored = MyModel()                                        # 待恢复参数的同一模型
checkpoint = tf.train.Checkpoint(myModel=model_to_be_restored)   # 键名保持为“myModel”

# 例：checkpoint.restore('./save/model.ckpt-1') 就可以载入前缀为 model.ckpt
# 当保存了多个文件时，我们往往想载入最近的一个。可以使用 tf.train.latest_checkpoint(save_path)
# 这个辅助函数返回目录下最近一次 checkpoint 的文件名。
# 例如如果 save 目录下有 model.ckpt-1.index 到 model.ckpt-10.index 的 10 个保存文件，
# tf.train.latest_checkpoint('./save') 即返回 ./save/model.ckpt-10 。
checkpoint.restore("save_path_with_prefix_and_index")


# checkpointManager
"""
    在长时间的训练后，程序会保存大量的 Checkpoint，但我们只想保留最后的几个 Checkpoint；
    Checkpoint 默认从 1 开始编号，每次累加 1，但我们可能希望使用别的编号方式（例如使用当前 Batch 的编号作为文件编号）。
"""
checkpoint = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(checkpoint, directory='./save', checkpoint_name='model.ckpt',
                                     max_to_keep=k)     # 保留的checkpoint数目

# 需要保存模型的时候，我们直接使用 manager.save()
manager.save()
# 如果我们希望自行指定保存的 Checkpoint 的编号
manager.save(checkpoint_number=100)































