#！-*- encoding=utf8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

""" 构建一个 1-10-1 的三层神经网络
"""

x = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0,0.02,x.shape)
y = np.square(x) + noise

# 定义两个placeholder (行不确定，列是1)
v1 = tf.placeholder(tf.float32, [None, 1])
v2 = tf.placeholder(tf.float32,[None,1])

# 定义神经网络中间层
weight_l1 = tf.Variable(tf.random_normal([1,10]))
biase_l1 = tf.Variable(tf.zeros([1,10]))
res_l1 = tf.matmul(v1,weight_l1)+biase_l1
l1 = tf.nn.tanh(res_l1)

# 定义神经网络输出层
weight_l2 = tf.Variable(tf.random_normal([10,1]))
biase_l2 = tf.Variable(tf.zeros([1,1]))
res_l2 = tf.matmul(l1,weight_l2) + biase_l2
predict = tf.nn.tanh(res_l2)

init = tf.global_variables_initializer()

# 定义损失函数为二次代价函数
loss = tf.reduce_mean(tf.square(v2 - predict))

# 定义优化器为梯度下降法,步长为 0.1
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(init)
    for i in range(5000):
        sess.run(train_step,feed_dict={v1:x,v2:y})
    # 获取预测值
    predict_value = sess.run(predict, feed_dict={v1: x})

    # 画图
    plt.scatter(x,y)
    plt.plot(x,predict_value,'r-')
    plt.show()


















