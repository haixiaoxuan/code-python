# *-* coding: utf-8 *-*

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

from datetime import datetime
import tensorflow as tf
from tensorflowonspark import TFNode


def print_log(worker_num, arg):
    print("{0}: {1}".format(worker_num, arg))


class ExportHook(tf.train.SessionRunHook):
    def __init__(self, mode, export_dir, input_tensor, output_tensor):
        self.mode = mode
        self.export_dir = export_dir
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def end(self, session):
        if self.mode != 'train':
            return

        print("{} ======= Exporting to: {}".format(datetime.now().isoformat(), self.export_dir))
        signatures = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: {
                'inputs': {'image': self.input_tensor},
                'outputs': {'prediction': self.output_tensor},
                'method_name': tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            }
        }

        # 保存和导出模型
        TFNode.export_saved_model(session,
                                  self.export_dir,
                                  tf.saved_model.tag_constants.SERVING,
                                  signatures)
        print("{} ======= Done exporting".format(datetime.now().isoformat()))


# 激活函数,默认 relu
def activation_fun(activation_option, input_maxtrix):
    if activation_option == 'ReLU':
        hid = tf.nn.relu(input_maxtrix)
    elif activation_option == 'ReLU6':
        hid = tf.nn.relu6(input_maxtrix)
    elif activation_option == 'Tanh':
        hid = tf.nn.tanh(input_maxtrix)
    elif activation_option == 'Sigmoid':
        hid = tf.nn.sigmoid(input_maxtrix)
    else:
        hid = tf.nn.relu(input_maxtrix)

    return hid


# 选择优化器，默认梯度下降
def optimize_fun(gradient_option, learning_rate):
    if gradient_option == 'GradientDescent':  # 包括：随机梯度SGD, 批梯度BGD, Mini批梯度MBGD
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif gradient_option == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=1)
    elif gradient_option == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif gradient_option == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif gradient_option == 'Adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
    elif gradient_option == 'Ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
    elif gradient_option == 'RMSProp':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    return optimizer


# 选择损失函数，默认sigmoid交叉熵
def loss_fun(loss_option, output, label):
    if loss_option == 'SigmoidCrossEntropy':
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=label))
    elif loss_option == 'L2':
        loss = tf.nn.l2_loss(label - output)
    elif loss_option == 'L1':
        loss = tf.reduce_mean(tf.abs(label - output))
    else:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=label))

    return loss


# 拆分batch，(x,y)
def feed_dict(batch , numpy):
    # Convert from [(images, labels)] to two numpy arrays of the proper type
    images = []
    labels = []
    for item in batch:
        images.append(item[0])
        labels.append(item[1])

    # 特征值
    xs = numpy.array(images)
    xs = xs.astype(numpy.float)

    # 标签值 (对于逻辑回归来说，标签值为int)
    ys = numpy.array(labels)
    ys = ys.astype(numpy.float)
    return (xs, ys)


# 拆分隐藏层参数 layers_active
def get_hidden_layer_info(arg):
    res = []
    if arg != None and len(arg) != 0:
        for i in arg.split("|"):
            tmp = []
            for j in i.split(","):
                tmp.append(j)
            res.append(tmp)
    return res

# 生成隐藏层
def generate_hidden_layer( in_x , in_count , out_count , name ,seed):
    weight = tf.Variable(tf.truncated_normal([in_count,out_count],seed=seed ),name = "hidden_weight_"+name)
    biase = tf.Variable(tf.random_normal([out_count],seed=seed),name="hidden_biase_"+name)
    return tf.nn.xw_plus_b(in_x ,weight,biase)


# tensorflow 训练/预测
def map_fun(args, ctx):
    import math
    import numpy
    import time

    worker_num = ctx.worker_num
    job_name = ctx.job_name
    task_index = ctx.task_index

    print_log(worker_num, args)

    # Delay PS nodes a bit, since workers seem to reserve GPUs more quickly/reliably (w/o conflict)
    if job_name == "ps":
        time.sleep((worker_num + 1) * 5)

    # Parameters
    FEATURE_COUNT = args.feature_count
    LABEL_COUNT = args.label_count
    BATCH_SIZE = args.batch_size
    SEED = args.seed

    # Get TF cluster and server instances
    cluster, server = ctx.start_cluster_server(1, args.rdma)

    if job_name == "ps":
        server.join()
    elif job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(  # 设置work资源
                worker_device="/job:worker/task:%d" % task_index,
                cluster=cluster)):

            # 定义 feature,lable 占位符
            with tf.name_scope('inputs'):
                x = tf.placeholder(tf.float32, [None, FEATURE_COUNT], name="x")
                y_target = tf.placeholder(tf.float32, [None, LABEL_COUNT], name="y_target")

            with tf.name_scope('layer'):

                # hidden layer
                layers_list = []  # 定义隐藏层每一层神经元的个数和输出
                layers_list.append([FEATURE_COUNT,x])
                hidden_info = get_hidden_layer_info(args.layers_active)
                if hidden_info != None and len(hidden_info) !=0 :
                    with tf.name_scope('hidden_layer'):
                        for index,lay in enumerate(hidden_info) :
                            h_out = generate_hidden_layer(layers_list[index][1], layers_list[index][0], int(lay[0]), str(index), SEED)
                            out = activation_fun(lay[1],h_out)
                            layers_list.append([int(lay[0]), out])

                # Variables of the output layer
                with tf.name_scope('output'):
                    # sm_w = tf.Variable(tf.truncated_normal([hidden_units, LABEL_COUNT], stddev=1.0 / math.sqrt(hidden_units)), name="sm_w")
                    out_info = layers_list.pop()
                    sm_w = tf.Variable(tf.truncated_normal(shape=[out_info[0], LABEL_COUNT],seed=SEED),
                                       name="sm_w")
                    sm_b = tf.Variable(tf.random_normal(shape=[LABEL_COUNT],seed=SEED), name="sm_b")
                    tf.summary.histogram("output_weights", sm_w)
                    y_output = tf.nn.softmax(tf.nn.xw_plus_b(out_info[1], sm_w, sm_b))  # 预测结果

            global_step = tf.train.get_or_create_global_step()  # 返回或创建一个全局步数的tensor

            with tf.name_scope('loss'):
                loss = loss_fun(args.loss, y_output, y_target)
                tf.summary.scalar("loss", loss)

            with tf.name_scope('train'):
                optimizer = optimize_fun(args.gradient, args.learning_rate)
                train_op = optimizer.minimize(loss, global_step=global_step)

            label = y_target

            # 计算准确率
            prediction_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(label,1), tf.argmax(y_output,1)), tf.float32))

            tf.summary.scalar("prediction_accuracy", prediction_accuracy)

            summary_op = tf.summary.merge_all()

        # 拿到模型保存路径
        logdir = ctx.absolute_path(args.model)
        logdir = args.model
        print("tensorflow model path: {0}".format(logdir))

        if job_name == "worker" and task_index == 0:
            summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

        hooks = [tf.train.StopAtStepHook(last_step=args.steps)] if args.mode == "train" else []

        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(task_index == 0),  # 从 task0 恢复
                                               checkpoint_dir=logdir,
                                               hooks=hooks,
                                               chief_only_hooks=[
                                                   ExportHook(args.mode, ctx.absolute_path(args.export_dir), x,
                                                              y_output)]) as mon_sess:
            step = 0
            tf_feed = ctx.get_data_feed(args.mode == "train")

            while not mon_sess.should_stop() and not tf_feed.should_stop():

                batch_xs, batch_ys = feed_dict(tf_feed.next_batch(BATCH_SIZE),numpy)
                feed = {x: batch_xs, y_target: batch_ys}

                if len(batch_xs) > 0:
                    if args.mode == "train":  # 训练
                        _, summary, step = mon_sess.run([train_op, summary_op, global_step], feed_dict=feed)
                        # print accuracy and save model checkpoint to HDFS every 100 steps
                        if (step % 100 == 0):
                            print("{0} step: {1} accuracy: {2}".format(datetime.now().isoformat(), step,
                                                                       mon_sess.run(prediction_accuracy,
                                                                                    {x: batch_xs, y_target: batch_ys})))

                        if task_index == 0:
                            summary_writer.add_summary(summary, step)

                    else:  # args.mode == "inference"
                        labels, preds, preds_acc = mon_sess.run([label, y_output, prediction_accuracy], feed_dict=feed)

                        # results = ["{0} Label: {1}, Prediction: {2}".format(datetime.now().isoformat(), l, p) for l, p in zip(labels, preds)]
                        results = ["{0},{1}".format(numpy.argmax(l), numpy.argmax(p)) for l, p in zip(labels, preds)]
                        # results = [p for p in preds]
                        print(results)
                        tf_feed.batch_results(results)

            if mon_sess.should_stop() or step >= args.steps:
                tf_feed.terminate()

        # Ask for all the services to stop.
        print("{0} stopping MonitoredTrainingSession".format(datetime.now().isoformat()))

    if job_name == "worker" and task_index == 0:
        summary_writer.close()

