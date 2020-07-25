#！-*-coding=utf8-*-

import tensorflow as tf
import numpy as np

x=np.random.rand(100)
y=x*0.3+0.8

# 构造一个线性模型
a=tf.Variable(0.)
b=tf.Variable(0.)
y_pre=a*x+b

# 二次代价函数
loss=tf.reduce_mean(tf.square(y-y_pre))
# 定义一个梯度下降法来进行训练的优化器,步长为 0.2
optimizer = tf.train.GradientDescentOptimizer(0.2)
# 使得 代价函数 最小
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # 迭代次数 200 次
    for i in range(200):
        sess.run(train)
        if i%20==0:
            print(i,sess.run([a,b]))





