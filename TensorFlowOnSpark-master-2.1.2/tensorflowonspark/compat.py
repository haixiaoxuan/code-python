# Copyright 2019 Yahoo Inc / Verizon Media
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.
"""Helper functions to abstract API changes between TensorFlow versions, intended for end-user TF code."""


"""
    适配 2.x 与 1.x 的一些方法
    
    主要方法：
        1. export_saved_model 持久化tf模型
        2. disable_auto_shard 关闭自动分片
        3. is_gpu_available 测试GPU是否可用
"""




import tensorflow as tf
from packaging import version


def export_saved_model(model, export_dir, is_chief=False):
  """ 保存模型 """
  if version.parse(tf.__version__) == version.parse('2.0.0'):
    if is_chief: # 如果is_chief 为 true， 则采用tf2.0的API进行导出
      tf.keras.experimental.export_saved_model(model, export_dir)
  else:
    model.save(export_dir, save_format='tf')


def disable_auto_shard(options):
  """ 分布式训练环境下， tensorflow默认会自动处理数据集分片 """
  if version.parse(tf.__version__) == version.parse('2.0.0'):
    options.experimental_distribute.auto_shard = False
  else:
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF


# 测试gpu是否可用
def is_gpu_available():
  if version.parse(tf.__version__) < version.parse('2.1.0'):
    return tf.test.is_built_with_cuda()
  else:
    return len(tf.config.list_physical_devices('GPU')) > 0
