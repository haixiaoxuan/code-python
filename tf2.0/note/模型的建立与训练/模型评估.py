import tensorflow as tf


# 该评估器能够对模型预测的结果与真实结果进行比较，并输出预测正确的样本数占总样本数的比例
# 我们迭代测试数据集，每次通过 update_state() 方法向评估器输入两个参数： y_pred 和 y_true
# （例如当前已传入的累计样本数和当前预测正确的样本数）
# 迭代结束后，我们使用 result() 方法输出最终的评估指标值
tf.keras.metrics.SparseCategoricalAccuracy


