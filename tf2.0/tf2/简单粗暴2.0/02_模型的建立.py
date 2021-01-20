import tensorflow as tf
import numpy as np

"""
        模型的构建： tf.keras.Model 和 tf.keras.layers
        
        模型的损失函数： tf.keras.losses        
                        sparse_categorical_crossentropy     label没有被one-hot编码时使用
                        categorical_crossentropy    使用时传入被one-hot编码的label -> tf.one_hot(y, depth=tf.shape(y_pred)[-1])
        模型的优化器： tf.keras.optimizer      Adam|
        
        模型的评估： tf.keras.metrics
                        SparseCategoricalAccuracy 
                        使用：
                            sparse_categorical_accuracy.update_state(y_true=y, y_pred=y_pred)
                            sparse_categorical_accuracy.result()        获取结果
        
        tf.keras.applications 中有一些预定义好的经典卷积神经网络结构，
                    如 VGG16 、 VGG19 、 ResNet 、 MobileNet 等。我们可以直接调用这些经典的卷积神经网络结构（甚至载入预训练的参数），
                    而无需手动定义网络结构。
                    model = tf.keras.applications.MobileNetV2()

"""


# 自定义模型，通过继承来实现
X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])


class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        """
            tf.keras.layers.Flatten()   将除batch_size之外的别的维度flat
            默认无激活函数，常用的激活函数包括 tf.nn.relu 、 tf.nn.tanh 和 tf.nn.sigmoid
            use_bias ：是否加入偏置向量 bias
            变量初始化：默认为 tf.glorot_uniform_initializer 
        """
        self.dense = tf.keras.layers.Dense(     # Dense全连接层 input = [batch_size, input_dim] -> 输出[batch_size, units]
            units=1,    # 表示输出张量
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, input):
        output = self.dense(input)
        return output


model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(X)      # 调用模型 y_pred = model(X) 而不是显式写出 y_pred = a * X + b
        loss = tf.reduce_mean(tf.square(y_pred - y))
    grads = tape.gradient(loss, model.variables)    # 使用 model.variables 这一属性直接获得模型中的所有变量
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
print(model.variables)



""" 
    迁移学习    model = tf.keras.applications.MobileNetV2()
        input_shape ：输入张量的形状（不含第一维的 Batch），大多默认为 224 × 224 × 3 。一般而言，模型对输入张量的大小有下限，长和宽至少为 32 × 32 或 75 × 75 ；
        include_top ：在网络的最后是否包含全连接层，默认为 True ；
        weights ：预训练权值，默认为 'imagenet' ，即为当前模型载入在 ImageNet 数据集上预训练的权值。如需随机初始化变量可设为 None ；
        classes ：分类数，默认为 1000。修改该参数需要 include_top 参数为 True 且 weights 参数为 None 。
    
"""



"""
    另外一种建立模型的方式：
        Functional API
"""
inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dense(units=10)(x)
outputs = tf.keras.layers.Softmax()(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


"""
    当模型建立完成后，通过 tf.keras.Model 的 compile 方法配置训练过程
        接受三个参数：
            oplimizer ：优化器，可从 tf.keras.optimizers 中选择；
            loss ：损失函数，可从 tf.keras.losses 中选择；
            metrics ：评估指标，可从 tf.keras.metrics 中选择。
    
    fit 接受五个参数：
        x ：训练数据；
        y ：目标数据（数据标签）；
        epochs ：将训练数据迭代多少遍；
        batch_size ：批次的大小；
        validation_data ：验证数据，可用于在训练过程中监控模型的性能。
"""



