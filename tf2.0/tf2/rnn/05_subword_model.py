import tensorflow_datasets as tfds

"""
    word level model       word词表一般很大，导致训练的模型很大，训练缓慢
    char level model       如果是英文的话就是26个字母加符号，
    subword level model    介于两者之间，采用 char组合方案，比如  hello -> he ll o
    
    
    import tensorflow_datasets as tfds
    这个库中有很多数据集，可以用来做测试

"""


# 数据下载到本地地址 C:\Users\xiexiaoxuan\tensorflow_datasets\imdb_reviews\subwords8k\1.0.0
dataset, info = tfds.load("imdb_reviews/subwords8k",
                          with_info=True,
                          as_supervised=True)   # 指有监督学习，是否返回label信息

train_dataset, test_dataset = dataset["train"], dataset["test"]


tokenizer = info.features["text"].encoder
print(tokenizer.vocab_size)     # 8185


# 样本示例，将 hello world 按照 subword进行编码
sample_string = "hello world"
tokenizer_string = tokenizer.encode(sample_string)
print(tokenizer_string)
original_string = tokenizer.decode(tokenizer_string)
print(original_string)


for token in tokenizer_string:
    print(token, tokenizer.decode([token]))


buffer_size = 10000
batch_size = 64
train_dataset = train_dataset.shuffle(buffer_size)

# 对batch进行填充，默认采用batch中的最长长度，也可以指定
train_dataset = train_dataset.padded_batch(batch_size, train_dataset.output_shapes)

