import tensorflow as tf
from tensorflow import keras


"""
    可以分为两种 one-hot| dense embedding
    
    训练embedding时因为句子单词个数不同，需要进行padding
        padding缺点：信息丢失，无效计算多
        所以RNN出现了
"""


imdb = keras.datasets.imdb
vocab_size = 10000      # 将词汇表数目限定到10000
index_from = 3      # 词表的index从几开始算
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size, index_from=index_from)


print(train_data[0], train_labels[0])
print(train_data.shape, train_labels.shape)     # 每一行都为变长数组，不确定其长度
print(len(train_data[0]), len(train_data[1]))


# 获取词汇表中单词索引
word_index = imdb.get_word_index()
print(len(word_index))
print(word_index)
word_index = {k: (v+3) for k, v in word_index.items()}           # 之前偏移3，所以现在要加上三


word_index['<PAD>'] = 0     # 填充字符
word_index['<START>'] = 1   # 起始字符
word_index['<UNK>'] = 2     # 缺失字符
word_index['<END>'] = 3     # 结束字符

# 反转为 index -> word
reverse_word_index = dict([(value, key) for key, value in word_index.items()])


def decode_review(text_ids):
    return ' '.join([reverse_word_index.get(word_id, "<UNK>") for word_id in text_ids])


# train_data 中的index 转变为具体单词
decode_review(train_data[0])


# 将长度低于500的句子补全，高于500的进行截断
max_length = 500
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,     # list of list
    value=word_index['<PAD>'],
    padding='post',         # post, pre  分别对应填充在前还是在后
    maxlen=max_length)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,  # list of list
    value=word_index['<PAD>'],
    padding='post',  # post, pre
    maxlen=max_length)
print(train_data[0])


# 开始进行 embedding
embedding_dim = 16      # 最终向量长度
batch_size = 128

model = keras.models.Sequential([
    # 1. define matrix: [vocab_size, embedding_dim]     词汇表中每个单词一个长度为16的向量
    # 2. [1,2,3,4..], max_length * embedding_dim        每一个样本都变成了 max_length * 16 的矩阵
    # 3. batch_size * max_length * embedding_dim        按批次输入训练 batch_size * max_length * embedding_dim
    keras.layers.Embedding(vocab_size, embedding_dim,
                           input_length=max_length),

    # batch_size * max_length * embedding_dim   将 max_length 这个维度消掉
    #   -> batch_size * embedding_dim
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, train_labels,
                    epochs=30,
                    batch_size=batch_size,
                    validation_split=0.2)
