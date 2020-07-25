import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

"""
    使用莎士比亚数据集来进行文本生成训练
"""


input_filepath = "./shakespeare.txt"
text = open(input_filepath, 'r').read()

print(len(text))
print(text[0:100])


# 1. generate vocab     # 生成词表
# 2. build mapping char->id     # 生成单词到 id 的映射
# 3. data -> id_data        # 将句子转为id
# 4. abcd -> bcd<eos>
vocab = sorted(set(text))       # 获取所有字符
print(len(vocab))
print(vocab)


# 构造映射关系
char2idx = {char: idx for idx, char in enumerate(vocab)}
print(char2idx)


idx2char = np.array(vocab)
print(idx2char)


# 将text转变为 id 列表
text_as_int = np.array([char2idx[c] for c in text])
print(text_as_int[0:10])
print(text[0:10])


def split_input_target(id_text):
    """
        构造训练数据， abcde -> abcd, bcde
    """
    return id_text[0:-1], id_text[1:]


# 构造字母个数为100的句子
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
seq_length = 100
seq_dataset = char_dataset.batch(seq_length + 1,
                                 drop_remainder=True)      # 此参数表示 是否丢掉最后一个不能构成batch的数据


for ch_id in char_dataset.take(2):
    print(ch_id, idx2char[ch_id.numpy()])

for seq_id in seq_dataset.take(2):
    print(seq_id)
    print(repr(''.join(idx2char[seq_id.numpy()])))      # repr 不显示特殊字符，如\n 就是换行


# 构造输入输出
seq_dataset = seq_dataset.map(split_input_target)
for item_input, item_output in seq_dataset.take(2):
    print(item_input.numpy())
    print(item_output.numpy())


batch_size = 64
buffer_size = 10000
seq_dataset = seq_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)


# 当作是分类，输出最终词的概率
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim,
                               batch_input_shape=[batch_size, None]),
        keras.layers.SimpleRNN(units=rnn_units,
                               stateful=True,
                               recurrent_initializer='glorot_uniform',
                               return_sequences=True),          # 输入输出都是序列，每一步的返回结果都作为输出
        keras.layers.Dense(vocab_size),     # 此处没有使用激活函数
    ])
    return model


model = build_model(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=batch_size)
model.summary()


for input_example_batch, target_example_batch in seq_dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape)      # (batch_size, 100, vocab_size)    vocab_size 是概率


# random sampling.  使用概率最大的生成文本即贪心策略greedy，或者随机生成
# greedy, random.
# logits 分类问题计算softmax值之前的输出就是 logits， num_samples 采样数为1
sample_indices = tf.random.categorical(logits=example_batch_predictions[0], num_samples=1)
print(sample_indices)       # (100, 65) -> (100, 1)


# 降低维度  (100, 1) -> (100)
sample_indices = tf.squeeze(sample_indices, axis=-1)
print(sample_indices)


print("Input: ", repr("".join(idx2char[input_example_batch[0]])))
print("Output: ", repr("".join(idx2char[target_example_batch[0]])))
print("Predictions: ", repr("".join(idx2char[sample_indices])))


def loss(labels, logits):
    return keras.losses.sparse_categorical_crossentropy(labels, logits,
                                                        from_logits=True)   # 因为输出层没有加激活函数，所以from_logits为True


model.compile(optimizer='adam', loss=loss)
example_loss = loss(target_example_batch, example_batch_predictions)
print(example_loss.shape)
print(example_loss.numpy().mean())


output_dir = "./text_generation_checkpoints"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
checkpoint_prefix = os.path.join(output_dir, 'ckpt_{epoch}')
checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

epochs = 10
history = model.fit(seq_dataset, epochs=epochs, callbacks=[checkpoint_callback])


tf.train.latest_checkpoint(output_dir)

model2 = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)        # 预测时一次只输入一个样本
model2.load_weights(tf.train.latest_checkpoint(output_dir))
model2.build(tf.TensorShape([1, None]))         # 指定模型输入数据shape， 一个样本, none表示输入可以是边长序列


# 文本生成流程
# A -> model -> b
# A.append(b) -> B
# B(Ab) -> model -> c
# B.append(c) -> C
# C(Abc) -> model -> ...
model2.summary()


def generate_text(model, start_string, num_generate=1000):
    input_eval = [char2idx[ch] for ch in start_string]
    input_eval = tf.expand_dims(input_eval, 0)          # 扩展维度，0

    text_generated = []
    model.reset_states()


    """
        temperature = 0.5
            使得概率变得越陡峭，随机值越趋近于最大值
    """
    temperature = 0.5

    for _ in range(num_generate):
        # 1. model inference -> predictions
        # 2. sample -> char -> text_generated.
        # 3. update input_eval

        # predictions : [batch_size, input_eval_len, vocab_size]
        predictions = model(input_eval)
        predictions = predictions / temperature
        # predictions : [input_eval_len, vocab_size]
        predictions = tf.squeeze(predictions, 0)
        # predicted_ids: [input_eval_len, 1]
        # a b c -> b c d
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()     # 只使用预测结果d
        text_generated.append(idx2char[predicted_id])
        # s, x -> rnn -> s', y
        # 此处需要注意，下次输入只是单个predicted_id, 并非text_generated，模型会保留上一步的状态来一起作为输入
        input_eval = tf.expand_dims([predicted_id], 0)

    return start_string + ''.join(text_generated)


new_text = generate_text(model2, "All: ")
print(new_text)

