
"""
    pc-01$ python example.py --job_name="ps" --task_index=0
    pc-02$ python example.py --job_name="worker" --task_index=0
    pc-03$ python example.py --job_name="worker" --task_index=1
    pc-04$ python example.py --job_name="worker" --task_index=2
"""



import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import time

cluster = tf.train.ClusterSpec({
    "worker": [
        "worker01.hadoop.dtmobile.cn:22222",  # 在代码中需要用这台机器计算的时候，就要定义：/job:worker/task:0
        "worker02.hadoop.dtmobile.cn:22222",  # /job:worker/task:1
        "worker03.hadoop.dtmobile.cn:22222"  # /job:worker/task:2
    ],
    "ps": [
        "master01.hadoop.dtmobile.cn:22222",  # /job:ps/task:0
    ]})

tf.app.flags.DEFINE_string("job_name", "", "ps|worker")
tf.app.flags.DEFINE_integer("task_index", 0, "task index")

FLAGS = tf.app.flags.FLAGS

server = tf.train.Server(cluster,
                         job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index)


# config
batch_size = 100
learning_rate = 0.0005
end_step = 10000
logs_path = "/tmp/mnist/1"      # summary

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":

    # 通过tf.train.replica_device_setter函数来指定执行每一个运算的设备
    # tf.train.replica_device_setter函数会自动将所有的参数分配到参数服务器上，而
    # 计算分配到当前的计算服务器上
    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):

        # count the number of updates
        global_step = tf.get_variable(
            'global_step',
            [],
            initializer=tf.constant_initializer(0),
            trainable=False)

        # input images
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
            y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

        # model parameters will change during training so we use tf.Variable
        tf.set_random_seed(1)
        with tf.name_scope("weights"):
            W1 = tf.Variable(tf.random_normal([784, 100]))
            W2 = tf.Variable(tf.random_normal([100, 10]))

        # bias
        with tf.name_scope("biases"):
            b1 = tf.Variable(tf.zeros([100]))
            b2 = tf.Variable(tf.zeros([10]))

        # implement model
        with tf.name_scope("softmax"):
            z2 = tf.add(tf.matmul(x, W1), b1)
            a2 = tf.nn.sigmoid(z2)
            z3 = tf.add(tf.matmul(a2, W2), b2)
            y = tf.nn.softmax(z3)

        # specify cost function
        with tf.name_scope('cross_entropy'):
            # this is our cost
            cross_entropy = tf.reduce_mean(
                -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

        # specify optimizer
        with tf.name_scope('train'):
            grad_op = tf.train.GradientDescentOptimizer(learning_rate)
            '''
            rep_op = tf.train.SyncReplicasOptimizer(
                grad_op,
                replicas_to_aggregate=len(workers),
                 replica_id=FLAGS.task_index, 
                 total_num_replicas=len(workers),
                 use_locking=True)
             train_op = rep_op.minimize(cross_entropy, global_step=global_step)
             '''
            train_op = grad_op.minimize(cross_entropy, global_step=global_step)

        '''
        init_token_op = rep_op.get_init_tokens_op()
        chief_queue_runner = rep_op.get_chief_queue_runner()
        '''

        with tf.name_scope('Accuracy'):
            # accuracy
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # create a summary for our cost and accuracy
        tf.summary.scalar("cost", cross_entropy)
        tf.summary.scalar("accuracy", accuracy)
        # merge all summaries into a single "operation" which we can execute in a session
        summary_op = tf.summary.merge_all()

        # 定义用于保存模型的saver
        saver = tf.train.Saver()

        # init parms
        init_op = tf.global_variables_initializer()
        print("Variables initialized ...")

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),  # 定义当前计算服务器是否为主计算服务器，只用主计算服务器会保存模型以及输出日志
                             logdir="/tmp/train_logs",  # 指定保存模型和输出日志的地址
                             global_step=global_step,  # 指定当前迭代的轮数，这个会用于生成保存模型文件的文件名
                             init_op=init_op,  # 指定初始化操作
                             summary_op=summary_op,  # 指定日志生成操作
                             saver=saver,  # 指定用于保存模型的saver
                             save_model_secs=60,  # 指定保存模型的时间间隔
                             save_summaries_secs=60  # 指定日志输出的时间间隔
                             )

    begin_time = time.time()
    frequency = 100

    # 通过设置log_device_placement选项来记录operations 和 Tensor 被指派到哪个设备上运行
    # 为了避免手动指定的设备不存在这种情况, 你可以在创建的 session 里把参数 allow_soft_placement
    # 设置为 True, 这样 tensorFlow 会自动选择一个存在并且支持的设备来运行 operation.
    # device_filters:硬件过滤器，如果被设置的话，会话会忽略掉所有不匹配过滤器的硬件。
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index]
    )

    with sv.prepare_or_wait_for_session(server.target, config=config) as sess:
        '''
        # is chief
        if FLAGS.task_index == 0:
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(init_token_op)
        '''
        # create log writer object (this will log on every machine)
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        # perform training cycles
        start_time = time.time()
        for epoch in range(10000):

            # 一个周期的批次数
            batch_count = int(mnist.train.num_examples / batch_size)

            for i in range(batch_count):
                batch_x, batch_y = mnist.train.next_batch(batch_size)

                # perform the operations we defined earlier on batch
                _, cost, summary, step = sess.run(
                    [train_op, cross_entropy, summary_op, global_step],
                    feed_dict={x: batch_x, y_: batch_y})
                writer.add_summary(summary, step)

                if step % frequency == 0 :
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print("Step: %d," % (step + 1),         # 全局步数
                          " Epoch: %d," % (epoch + 1),      # 局部epoch
                          " Cost: %.4f," % cost,            # 交叉熵损失函数
                          )

            if step >= end_step:
                break

        print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        print("Total Time: %3.2fs" % float(time.time() - begin_time))
        print("Final Cost: %.4f" % cost)

    sv.stop()
    print("done")



# if __name__ == "__main__":
#     # 调用main函数，并且构造 FLAG，也可以 tf.app.run(test)
#     tf.app.run()
