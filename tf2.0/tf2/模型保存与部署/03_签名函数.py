import tensorflow as tf


"""
    将 tf.function 标注的函数保存为 saved_model
"""


@tf.function(input_signature=[tf.TensorSpec([None], tf.int32, name="x")])
def cube(z):
    return tf.pow(z, 3)


# 转变为 Concrete_function 具体函数
cube_concrete = cube.get_concrete_function(tf.TensorSpec([None], tf.int32))


# 保存
to_export = tf.Module()
to_export.cube = cube   # 给model添加成员变量
tf.saved_model.save(to_export, "./signature_to_saved_model")


# > saved_model_cli show --dir ./signature_to_saved_model --all
load = tf.saved_model.load("./signature_to_saved_model")
res = load.cube(tf.constant([9]))
print(res.numpy())












