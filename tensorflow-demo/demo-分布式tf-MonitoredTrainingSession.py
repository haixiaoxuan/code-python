
"""
    python trainer.py \
    --ps_hosts=192.168.0.203:2222 \
    --worker_hosts=192.168.0.205:2222,192.168.0.206:2222 \
    --job_name=ps \
    --task_index=0

    python trainer.py \
    --ps_hosts=192.168.0.203:2222 \
    --worker_hosts=192.168.0.205:2222,192.168.0.206:2222 \
    --job_name=worker \
    --task_index=0

    python trainer.py \
    --ps_hosts=192.168.0.203:2222 \
    --worker_hosts=192.168.0.205:2222,192.168.0.206:2222 \
    --job_name=worker
    --task_index=1


    Estimator API
    cluster = {'ps': ['10.134.96.44:2222', '10.134.96.184:2222'],
           'worker': ['10.134.96.37:2222', '10.134.96.145:2222']}
    os.environ['TF_CONFIG'] = json.dumps(
        {'cluster': cluster,
         'task': {'type': 'worker', 'index': 0}})


"""
import argparse
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None


# 定义一个函数，用于初始化所有的权值 W
def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


# 定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


# 定义一个函数，用于构建卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


# 定义一个函数，用于构建池化层
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index,
                             start=True)

    # 参数服务器只需要管理TensorFlow中的变量，不需要执行训练的过程。
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        # 通过tf.train.replica_device_setter函数来指定执行每一个运算的设备
        # tf.train.replica_device_setter函数会自动将所有的参数分配到参数服务器上，而
        # 计算分配到当前的计算服务器上

        # tf.train.replica_device_setter()会根据job名，将with内的Variable op放到ps tasks，
        # 将其他计算op放到worker tasks。默认分配策略是轮询。
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            x = tf.placeholder(tf.float32, [None, 784])
            y_actual = tf.placeholder(tf.float32, [None, 10])
            keep_prob = tf.placeholder(tf.float32)

            # 重构 input,因为卷积层和池化层要求x为 1行4列的矩阵，而输入input为1行2列
            # [训练集数量， 图像高， 图像宽， 通道数量]
            x_image = tf.reshape(x, [-1, 28, 28, 1])

            # 构建网络

            # 第一层卷积, [卷积核高， 卷积核宽， 输入通道数， 输出通道数]
            W_conv1 = weight_variable([5, 5, 1, 32])
            b_conv1 = bias_variable([32])
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 第一个卷积层
            h_pool1 = max_pool(h_conv1)  # 第一个池化层,[-1, 14, 14, 32]

            # 第二层卷积
            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 第二层卷积层
            h_pool2 = max_pool(h_conv2)  # 第二层池化层,[-1, 7, 7, 64]

            # 第一层全连接层
            W_fc1 = weight_variable([7 * 7 * 64, 1024])
            b_fc1 = bias_variable([1024])
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # reshape成向量
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 全连接层

            # dropout
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # dropout

            # 第二层全连接，分类 softmax
            W_fc2 = weight_variable([1024, 10])
            b_fc2 = bias_variable([10])
            y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # softmax, [-1, 10]

            # 最优化求解
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual * tf.log(y_predict), 1))  # 交叉熵，没有平均值
            global_step = tf.train.get_or_create_global_step()
            optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
            train_op = optimizer.minimize(cross_entropy, global_step=global_step)

            cross_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))
            accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32))  # 精度计算

            # tensorboard
            tf.summary.scalar('cost', cross_entropy)
            tf.summary.scalar("accuracy", accuracy)
            summary_op = tf.summary.merge_all()

            # The StopAtStepHook handles stopping after running given steps.
            hooks = [tf.train.StopAtStepHook(last_step=400)]

            # 通过设置log_device_placement选项来记录operations 和 Tensor 被指派到哪个设备上运行
            # 为了避免手动指定的设备不存在这种情况, 你可以在创建的 session 里把参数 allow_soft_placement
            # 设置为 True, 这样 tensorFlow 会自动选择一个存在并且支持的设备来运行 operation.
            # device_filters:硬件过滤器，如果被设置的话，会话会忽略掉所有不匹配过滤器的硬件。
            # config = tf.ConfigProto(
            #     allow_soft_placement=True,
            #     log_device_placement=False,
            #     device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index]
            # )

            # 通过设置log_device_placement选项来记录operations 和 Tensor 被指派到哪个设备上运行
            config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index]
            )

            # The MonitoredTrainingSession takes care of session initialization,
            # restoring from a checkpoint, saving to a checkpoint, and closing when done
            # or an error occurs.
            # master="grpc://" + worker_hosts[FLAGS.task_index]
            # if_chief: 制定task_index为0的任务为主任务，用于负责变量初始化、做checkpoint、保存summary和复原
            # 定义计算服务器需要运行的操作。在所有的计算服务器中有一个是主计算服务器。
            # 它除了负责计算反向传播的结果，它还负责输出日志和保存模型
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   config=config,
                                                   is_chief=(FLAGS.task_index == 0),
                                                   hooks=hooks,
                                                   checkpoint_dir=FLAGS.checkpoint_dir) as mon_sess:
                while not mon_sess.should_stop():
                    # Run a training step asynchronously.
                    # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                    # perform *synchronous* training.
                    # mon_sess.run handles AbortedError in case of preempted PS.
                    batch_x, batch_y = mnist.train.next_batch(64)
                    step, _ = mon_sess.run([global_step, train_op], feed_dict={
                        x: batch_x,
                        y_actual: batch_y,
                        keep_prob: 0.8})

                    if step > 0 and step % 10 == 0:
                        loss, acc = mon_sess.run([cross_entropy, accuracy], feed_dict={
                            x: batch_x,
                            y_actual: batch_y,
                            keep_prob: 1.0})
                        print("step: %d, loss: %f, acc: %f" % (step, loss, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument("--ps_hosts", type=str, default="", help="Comma-separated list of hostname:port pairs")
    parser.add_argument("--worker_hosts", type=str, default="", help="Comma-separated list of hostname:port pairs")
    parser.add_argument("--job_name", type=str, default="", help="One of 'ps', 'worker'")
    # Flags for defining the tf.train.Server
    parser.add_argument("--task_index", type=int, default=0, help="Index of task within the job")

    parser.add_argument("--checkpoint_dir", type=str, default=None, help="path to a directory where to restore variables.")

    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")

    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)
