#!-*-coding=utf8-*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

""" 最好不要在本机运行，会死机 """

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

batch_size = 100
batch_num = mnist.train.num_examples // batch_size


def weight_variable(shape):
    """ 初始化权重 """
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """ 初始化bias """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """
    卷积层
    :param x: input tensor of shape [batch,in_height,in_width,in_channels]
    :param W: filter/kernel tensor of shape [filter_height,filter_width,in_channels,out_channels]
    :return     strides[0] = strides[3] = 1,strides[1]代表x方向步长，strides[2]代表y方向的步长
                padding -> str    SAME | VALID
                    SAME 输入图像和输出图像一样大，除以步长向上取整
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    """
    池化层
    :param x:
    :return:   ksize [1,x,y,1]
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 改变 x 格式 为 [batch,in_height,in_width,in_channels]
x_imag = tf.reshape(x, [-1, 28, 28, 1])

# 初始化第一个卷积层的权重和偏度值
W_conv1 = weight_variable([3, 3, 1, 32])  # 5*5 的filter，32个filter，从 1 个平面抽取特征
b_conv1 = bias_variable([32])  # 每个卷积核一个偏度值

# 进行卷积操作，然后加上偏置值，再应用于relu激活函数
h_conv1 = tf.nn.relu(conv2d(x_imag, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  # 池化

# 第二层卷积和池化
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 第三层卷积和池化
W_conv3 = weight_variable([3, 3, 64, 128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# 经过上面的操作变成了64张 7*7 的平面, 构建全连接层
W_fc1 = weight_variable([4 * 4 * 128, 1024])
b_fc1 = bias_variable([1024])

# 将卷积池化后的结果变为一维
h_pool2_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 使用keep_prob 来表示神经元的输出概率
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第二层全连接
W_fc2 = weight_variable([1024, 1024])
b_fc2 = bias_variable([1024])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# 第二层全连接
W_fc3 = weight_variable([1024, 10])
b_fc3 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_arr = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_arr, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(10000):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.7})

        if epoch % 100 == 0:
            acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1})
            print("step {0}  accuracy:{1}".format(epoch, acc))

    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1})
    print("epoch: " + str(epoch) + "  accuracy: " + str(acc))

