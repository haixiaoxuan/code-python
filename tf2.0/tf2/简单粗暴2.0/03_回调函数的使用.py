import os
from tensorflow import keras
import tensorflow as tf


"""
    EarlyStopping
        参数1：monitor 字符串类型 -> acc val_acc loss val_loss
            监控数据的类型
        参数2：min_delta 
            增大或者减小的阈值，只有大于这个部分才算作improvement
        参数3：patience
            能够容忍多少个epoch内都没有improvement
        参数4：mode 字符串类型 -> auto min max
            控制上升还是下降
             
"""


# 定义回调函数
# tensorboard   earlystopping  modelcheckpoint
logdir = os.path.join("callbacks")
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir, "checkpoint_fashion_mnist.h5")

call_backs = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True),    # 保存最好的模型，否则保存最新的模型
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3),
]


history = model.fit(x_train_scaled, y_train, epochs=10, validation_data=(x_valid_scaled, y_valid), callbacks=call_backs)


"""
    单独的tensorboard使用
    
"""
summary_writer = tf.summary.create_file_writer('./tensorboard')     # 参数为记录文件所保存的目录
# 开始模型训练
for batch_index in range(num_batches):
    # ...（训练代码，当前batch的损失值放入变量loss中）
    with summary_writer.as_default():                               # 希望使用的记录器
        tf.summary.scalar("loss", loss, step=batch_index)
        tf.summary.scalar("MyScalar", my_scalar, step=batch_index)  # 还可以添加其他自定义的变量


"""
    在tensorboard中保存graph 和 profile信息
        如计算图的结构，每个操作所耗费的时间等
    只需要在训练完成后一次写入即可
"""
tf.summary.trace_on(graph=True, profiler=True)  # 开启Trace，可以记录图结构和profile信息
# 进行训练
with summary_writer.as_default():
    tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)    # 保存Trace信息到文件

















