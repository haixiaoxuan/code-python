import tensorflow as tf
from tensorflow import keras
import os

"""
    1. 将 keras model 转换为 estimator model
    2. 预定义 estimator model 使用
"""



estimator = keras.estimator.model_to_estimator(model)
# 1.input_fn -> function
# 2.return a.(feature,labels) b.dataset->(feature,label)
estimator.train(input_fn=lambda: make_dataset(train_df, y_train, epochs=100))





"""
    预定义 estimator
        BaselineClassifier
        LinearClassifier
        DNNClassifier
        
    准备步骤：
        编写一个或多个数据集导入函数
        定义特征列
"""


def make_dataset(data_df, label_df, epochs=10, shuffle=True, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset


feature_column = tf.feature_column.numeric_column("age")
feature_columns = [feature_column]

# BaselineClassifier
output_dir = 'baseline_nodel'
baseline_estimator = tf.estimator.BaselineClassifier(model_dir=output_dir, n_classes=2)
baseline_estimator.train(input_fn=lambda: make_dataset(train_df, y_train, epochs=100))
baseline_estimator.evaluate(input_fn=lambda: make_dataset(eval_df, y_eval, epochs=1, shuffle=False, batch_size=20))


# LinearClassifier      feature_columns 为 [tf.feature_column 中]
liner_output_dir = 'linear_model'
line_estimator = tf.estimator.LinearClassifier(model_dir=liner_output_dir, n_classes=2, feature_columns=feature_columns)
line_estimator.train(input_fn=lambda: make_dataset(train_df, y_train, epochs=100))
line_estimator.evaluate(input_fn=lambda: make_dataset(eval_df, y_eval, epochs=1, shuffle=False))


# DNNClassifier
dnn_output_dir = './dnn_model'
dnn_estimator = tf.estimator.DNNClassifier(
    model_dir=dnn_output_dir, n_classes=2,
    feature_columns=feature_columns, hidden_units=[218, 128],
    activation_fn=tf.nn.relu, optimizer='Adam')
dnn_estimator.train(input_fn=lambda: make_dataset(train_df,y_train,epochs=100))
dnn_estimator.evaluate(input_fn=lambda: make_dataset(eval_df, y_eval, epochs=1, shuffle=False))



















