import tensorflow as tf
from tensorflow import keras


"""
    RNN
"""



model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim,
                           input_length=max_length),

    # 单层单向RNN
    keras.layers.SimpleRNN(units=64,
                           return_sequences=False       # 使用最后的输出为总输出
                           ),

    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])



model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim,
                           input_length=max_length),

    # 双层双向RNN
    keras.layers.Bidirectional(keras.layers.SimpleRNN(units=64, return_sequences=True)),    # 因为下一层也是RNN，所以下一层的输入也应该时一个序列
    keras.layers.Bidirectional(keras.layers.SimpleRNN(units=64, return_sequences=False)),

    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])

















