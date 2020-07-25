import tensorflow as tf
from tensorflow import keras
import numpy as np


"""
    LSTM
        还可以添加如下参数：,这两个参数作用比较大，这两个参数在普通RNN中也有
            stateful=True
            recurrent_initializer="glorot_uniform"
"""



model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim,
                           input_length=max_length),

    # 单层单向RNN
    keras.layers.LSTM(units=64,
                           return_sequences=False       # 使用最后的输出为总输出
                           ),

    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])



model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim,
                           input_length=max_length),

    # 双层双向RNN
    keras.layers.Bidirectional(keras.layers.LSTM(units=64, return_sequences=True)),    # 因为下一层也是RNN，所以下一层的输入也应该时一个序列
    keras.layers.Bidirectional(keras.layers.LSTM(units=64, return_sequences=False)),

    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])



"""
    简单粗暴 tf2.0
"""


class DataLoader():
    """ 将文本中的所有字母符号进行数字编码 """
    def __init__(self):
        path = tf.keras.utils.get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        with open(path, encoding='utf-8') as f:
            self.raw_text = f.read().lower()
        self.chars = sorted(list(set(self.raw_text)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.text = [self.char_indices[c] for c in self.raw_text]

    def get_batch(self, seq_length, batch_size):
        seq = []
        next_char = []
        for i in range(batch_size):
            index = np.random.randint(0, len(self.text) - seq_length)
            seq.append(self.text[index:index+seq_length])
            next_char.append(self.text[index+seq_length])
        return np.array(seq), np.array(next_char)


"""
    在 __init__ 方法中我们实例化一个常用的 LSTMCell 单元，以及一个线性变换用的全连接层，我们首先对序列进行 “One Hot” 操作，(也可以进行deep embedding)
    即将序列中的每个字符的编码 i 均变换为一个 num_char 维向量，其第 i 位为 1，其余均为 0。变换后的序列张量形状为 [seq_length, num_chars] 。
    然后，我们初始化 RNN 单元的状态，存入变量 state 中。接下来，将序列从头到尾依次送入 RNN 单元，即在 t 时刻，
    将上一个时刻 t-1 的 RNN 单元状态 state 和序列的第 t 个元素 inputs[t, :] 送入 RNN 单元，得到当前时刻的输出 output 和 RNN 单元状态。
    取 RNN 单元最后一次的输出，通过全连接层变换到 num_chars 维，即作为模型的输出。
"""

class RNN(tf.keras.Model):
    def __init__(self, num_chars, batch_size, seq_length):
        super().__init__()
        self.num_chars = num_chars
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.cell = tf.keras.layers.LSTMCell(units=256)
        self.dense = tf.keras.layers.Dense(units=self.num_chars)

    def call(self, inputs, from_logits=False):
        inputs = tf.one_hot(inputs, depth=self.num_chars)       # [batch_size, seq_length, num_chars]
        state = self.cell.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        for t in range(self.seq_length):
            output, state = self.cell(inputs[:, t, :], state)
        logits = self.dense(output)
        if from_logits:
            return logits
        else:
            return tf.nn.softmax(logits)


num_batches = 1000
seq_length = 40
batch_size = 50
learning_rate = 1e-3

data_loader = DataLoader()
model = RNN(num_chars=len(data_loader.chars), batch_size=batch_size, seq_length=seq_length)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(seq_length, batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))


"""
    关于文本生成的过程有一点需要特别注意。之前，我们一直使用 tf.argmax() 函数，将对应概率最大的值作为预测值。
    然而对于文本生成而言，这样的预测方式过于绝对，会使得生成的文本失去丰富性。于是，我们使用 np.random.choice() 函数按照生成的概率分布取样。
    这样，即使是对应概率较小的字符，也有机会被取样到。同时，我们加入一个 temperature 参数控制分布的形状，参数值越大则分布越平缓（最大值和最小值的差值越小），
    生成文本的丰富度越高；参数值越小则分布越陡峭，生成文本的丰富度越低。
"""


# 此处应该是model中的方法
def predict(self, inputs, temperature=1.):
    batch_size, _ = tf.shape(inputs)
    logits = self(inputs, from_logits=True)
    prob = tf.nn.softmax(logits / temperature).numpy()
    return np.array([np.random.choice(self.num_chars, p=prob[i, :])
                     for i in range(batch_size.numpy())])


X_, _ = data_loader.get_batch(seq_length, 1)
for diversity in [0.2, 0.5, 1.0, 1.2]:
    X = X_
    print("diversity %f:" % diversity)
    for t in range(400):
        y_pred = model.predict(X, diversity)
        print(data_loader.indices_char[y_pred[0]], end='', flush=True)
        X = np.concatenate([X[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)
    print("\n")
















