import tensorflow_datasets as tfds
import tensorflow as tf

tfds.disable_progress_bar()
import os

"""
    这是在一台计算机上的多 GPU（单机多卡）进行同时训练的图形内复制（in-graph replication）。
    事实上，它会将所有模型的变量复制到每个GPU上，然后，通过使用 all-reduce 去整合所有处理器的梯度（gradients），
    并将整合的结果应用于所有副本之中。
        
        1. 所有变量和模型图都复制在副本上。
        2. 输入都均匀分布在副本中。
        3. 每个副本在收到输入后计算输入的损失和梯度。
        4. 通过求和，每一个副本上的梯度都能同步。
        5. 同步后，每个副本上的复制的变量都可以同样更新。
        
    如果需要禁用GPU
    在代码中设置：os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    note:
        自定义 call_back 中的方法(printLR, 每个epochs 结束后打印 学习率)
    
    参考：https://tensorflow.google.cn/tutorials/distribute/custom_training?hl=zh-cn
"""

# 将 with_info 设置为 True 会包含整个数据集的元数据,其中这些数据集将保存在 info
datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
mnist_train, mnist_test = datasets['train'], datasets['test']

# 自定义分配策略
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# 设置输入管道
# 通常来说，使用适合 GPU 内存的最大批量大小（batch size），并相应地调整学习速率。
num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples
print("训练数据集大小: {0} 测试数据集大小: {1}".format(num_train_examples, num_test_examples))

BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync


def scale(image, label):
    """ 完成归一化 """
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

# 生成模型（在strategy.scope上下文中）
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

# 定义检查点（checkpoint）目录以存储检查点（checkpoints）
checkpoint_dir = './training_checkpoints'
# 检查点（checkpoint）文件的名称
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")


# 衰减学习率的函数。 您可以定义所需的任何衰减函数。
def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5


# 在每个 epoch 结束时打印LR的回调（callbacks）。
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(epoch + 1, model.optimizer.lr.numpy()))


callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]

# 训练和评估
model.fit(train_dataset, epochs=12, callbacks=callbacks)

# 评估
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
eval_loss, eval_acc = model.evaluate(eval_dataset)
print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))

# 如要使用tensorboard > tensorboard --logdir=path/to/log-directory

# 导出模型 将图形和变量导出为与平台无关的 SavedModel 格式。 保存模型后，可以在有或没有 scope 的情况下加载模型。
path = 'saved_model/'
tf.keras.experimental.export_saved_model(model, path)

# 在无需 strategy.scope 加载模型
unreplicated_model = tf.keras.experimental.load_from_saved_model(path)
unreplicated_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy'])
eval_loss, eval_acc = unreplicated_model.evaluate(eval_dataset)
print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))

# 在含 strategy.scope 加载模型。
with strategy.scope():
    replicated_model = tf.keras.experimental.load_from_saved_model(path)
    replicated_model.compile(loss='sparse_categorical_crossentropy',
                             optimizer=tf.keras.optimizers.Adam(),
                             metrics=['accuracy'])

    eval_loss, eval_acc = replicated_model.evaluate(eval_dataset)
    print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))
