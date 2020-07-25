import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import tensorflowjs as tfjs


"""
    https://tf.wiki/zh/deployment/lite.html

    TensorFlow Lite 是 TensorFlow 在移动和 IoT 等边缘设备端的解决方案，提供了 Java、Python 和 C++ API 库
    目前 TFLite 只提供了推理功能，在服务器端进行训练后，经过如下简单处理即可部署到边缘设备上。


    tf Lite Converter
    
"""

loaded_model = keras.models.load_model('./graph_def_and_weights/fashion_mnist_model.h5')
tfjs.converters.save_keras_model(loaded_model, "./tfjs_models/keras_to_tfjs_layers_py/")

# 命令行下运行
# > tensorflowjs_converter --help
# > tensorflowjs_converter --input_format keras \
#     --output_format tfjs_layers_model \
#     ./graph_def_and_weights/fashion_mnist_model.h5 \
#     ./tfjs_models/keras_to_tfjs_layers

# > tensorflowjs_converter --input_format tf_saved_model \
#     --output_format tfjs_graph_model \
#     ./keras_saved_graph/ \
#     ./tfjs_models/saved_model_to_tfjs_model

# > tensorflowjs_converter --input_format tf_saved_model \
#     --output_format tfjs_layers_model \
#     ./keras_saved_graph/ \
#     ./tfjs_models/saved_model_to_tfjs_layers




"""
    tf Lite 量化
        如果不需要量化，去掉中间那一行即可，即 optimizations
"""

# keras_model -> tf_lite
loaded_keras_model = keras.models.load_model('./graph_def_and_weights/fashion_mnist_model.h5')
loaded_keras_model(np.ones((1, 28, 28)))

keras_to_tflite_converter = tf.lite.TFLiteConverter.from_keras_model(loaded_keras_model)
keras_to_tflite_converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
keras_tflite = keras_to_tflite_converter.convert()

if not os.path.exists('./tflite_models'):
    os.mkdir('./tflite_models')
with open('./tflite_models/quantized_keras_tflite', 'wb') as f:
    f.write(keras_tflite)


# concrete_function -> tf_lite
run_model = tf.function(lambda x: loaded_keras_model(x))
keras_concrete_func = run_model.get_concrete_function(tf.TensorSpec(loaded_keras_model.inputs[0].shape,
                                                                    loaded_keras_model.inputs[0].dtype))

concrete_func_to_tflite_converter = tf.lite.TFLiteConverter.from_concrete_functions([keras_concrete_func])
concrete_func_to_tflite_converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
concrete_func_tflite = concrete_func_to_tflite_converter.convert()

with open('./tflite_models/quantized_concrete_func_tflite', 'wb') as f:
    f.write(concrete_func_tflite)


# saved_model -> tf_lite
saved_model_to_tflite_converter = tf.lite.TFLiteConverter.from_saved_model('./keras_saved_graph/')
saved_model_to_tflite_converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
saved_model_tflite = saved_model_to_tflite_converter.convert()

with open('./tflite_models/quantized_saved_model_tflite', 'wb') as f:
    f.write(saved_model_tflite)




"""
    TFLite Interpreter
"""

with open('./tflite_models/concrete_func_tflite', 'rb') as f:
    concrete_func_tflite = f.read()

interpreter = tf.lite.Interpreter(model_content=concrete_func_tflite)
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)


input_data = tf.constant(np.ones(input_details[0]['shape'], dtype=np.float32))
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_results = interpreter.get_tensor(output_details[0]['index'])
print(output_results)






