import numpy as np
from tensorflow import keras
import tensorflow as tf

"""
    将keras model 转为concrete_function 具体函数
"""


loaded_keras_model = keras.models.load_model('./graph_def_and_weights/fashion_mnist_model.h5')
loaded_keras_model(np.ones((1, 28, 28)))


run_model = tf.function(lambda x: loaded_keras_model(x))
keras_concrete_func = run_model.get_concrete_function(tf.TensorSpec(loaded_keras_model.inputs[0].shape,
                                                                    loaded_keras_model.inputs[0].dtype))
keras_concrete_func(tf.constant(np.ones((1, 28, 28), dtype=np.float32)))


# saved_model -> concrete_function
# 参考 02_saved_model











