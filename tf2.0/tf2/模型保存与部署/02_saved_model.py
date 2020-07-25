import tensorflow as tf


"""
    保存 saved_model, 即 pb 格式
        SavedModel 包含了一个 TensorFlow 程序的完整信息： 不仅包含参数的权值，还包含计算的流程（即计算图）
        
    note:
        对于使用继承 tf.keras.Model 类建立的 Keras 模型 model ，使用 SavedModel 载入后将无法使用 model() 直接进行推断，而需要使用 model.call() 。
        使用继承 tf.keras.Model 类建立的 Keras 模型同样可以以相同方法导出，唯须注意 call 方法需要以 @tf.function 修饰，以转化为 SavedModel 支持的计算图
"""

tf.saved_model.save(model, "path")


# 命令行, 查看模型签名信息
# > saved_model_cli show --dir path --all


# 使用命令行来检查模型
# > saved_model_cli run --dir path --tag_set server --signature_def serving_default
# > --input_exprs 'flatten_input=np.ones((2, 28, 28))'


# 使用程序来进行加载模型
model = tf.saved_model.load("path")
# 查看签名信息
model.signatures.keys()


# 通过签名信息获取 ConcreteFunction， 即具体函数
inference = model.signatures['签名']


# 查看具体函数 output
print(inference.structured_outputs)
# 使用具体函数进行推理
results = inference(tf.constant(x_test[0:1]))
print(results['dense_2'])




