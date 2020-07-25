from tensorflow import keras


"""
    # 如果刚开始loss没有明显下降，训练一段时间后开始下降：

    # 深度神经网络参数众多，训练不充分
    # 梯度消失 -> 层数太深引起的梯度消失
    # 批归一化在以一定程度上可以缓解梯度消失问题
    # selu 激活函数可以自带归一化，并且效果要比 batchNormalization 效果要好，训练更快
    
"""


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
for _ in range(30):
    model.add(keras.layers.Dense(100, activation="selu"))

for _ in range(30):
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.BatchNormalization())
    """
        添加批归一化：这里还有一种方式将批归一化添加到激活函数前
        model.add(keras.layers.Dense(100))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))
    """




# 添加dropout层，一般只在最后几层添加
# model.add(keras.layers.Dropout(rate=0.5))
"""
    在训练时，每个神经单元以概率p被保留(dropout丢弃率为1-p)
    Alpha Dropout是一种保持输入均值和方差不变的Dropout，该层的作用是通过缩放和平移使得在dropout时也保持数据的自规范性。
    Alpha Dropout与SELU激活函数配合较好

    AlphaDoupout
        1. 均值和方差不变
        2. 归一化性质也不变
"""
model.add(keras.layers.AlphaDropout(rate=0.5))



model.add(keras.layers.Dense(10, activation="softmax"))

# 如果y已经经过one-hot编码，则直接使用 categorical_crossentropy 损失函数
# metrics 表示要统计什么指标
model.compile(loss="sparse_categorical_crossentropy",
             optimizer="sgd",
             metrics=["accuracy"])



