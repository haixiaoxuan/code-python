# -*-coding=utf8-*-

import tensorflow as tf
from tensorflowonspark import TFNode
import numpy
from algorithm_utils import tensorflow_utils
from algorithm_utils import constants
from threading import Thread
from datetime import datetime
import time


def save_model(sess, args, x, prediction):
    """ 保存模型 """

    pb_folder_dir = args.export_dir + constants.PATH_SEP + constants.PB_FOLDER_NAME
    # exported signatures defined in code
    signatures = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: {
            "inputs": {constants.SIG_INPUT: x},
            "outputs": {constants.SIG_OUTPUT: prediction},
            "method_name": tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        }
    }
    TFNode.export_saved_model(sess,
                              pb_folder_dir,
                              tf.saved_model.tag_constants.SERVING,
                              signatures)

    # 转为单个pb
    t = Thread(target=tensorflow_utils.convert_as_single_pb,
               args=[pb_folder_dir,
                     constants.PREDICT_NODE_NAME,
                     args.export_dir + constants.PATH_SEP + constants.PB_NAME])
    t.start()
    t.join()


def feed_dict(batch, args):
    """ """
    # Convert from dict of named arrays to two numpy arrays of the proper type
    images = batch[args.feature_alias]
    labels = batch[args.label_name]
    xs = numpy.array(images)
    xs = xs.astype(numpy.float32)
    ys = numpy.array(labels)
    ys = ys.astype(numpy.float32)
    return (xs, ys)


def map_fun(args, ctx):

    worker_num = ctx.worker_num
    job_name = ctx.job_name
    task_index = ctx.task_index

    # Parameters
    FEATURE_COUNT = args.feature_count
    LABEL_COUNT = args.label_count
    BATCH_SIZE = args.batch_size
    SEED = args.seed

    # Delay PS nodes a bit, since workers seem to reserve GPUs more quickly/reliably (w/o conflict)
    if job_name == "ps":
        time.sleep((worker_num + 1) * 5)

    # Get TF cluster and server instances
    cluster, server = TFNode.start_cluster_server(ctx, 1, args.protocol == "rdma")


    if job_name == "ps":
        server.join()
    elif job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index,
                cluster=cluster)):

            # Placeholders or QueueRunner/Readers for input data
            with tf.name_scope(constants.INPUT_LAYER_NAME):
                x = tf.placeholder(tf.float32, [None, FEATURE_COUNT], name=constants.INPUT_LAYER_X)
                y_ = tf.placeholder(tf.float32, [None, LABEL_COUNT], name=constants.INPUT_LAYER_Y)

            with tf.name_scope("layer"):

                # hidden layer
                n_count, h_out = tensorflow_utils.auto_generate_hidden_layer(FEATURE_COUNT, x, args.layers_active, SEED)

                # Variables of the output layer
                with tf.name_scope("output"):

                    sm_w = tf.Variable(tf.truncated_normal(shape=[n_count, LABEL_COUNT],
                                                           stddev=0.1, seed=SEED), name="sm_w")
                    sm_b = tf.Variable(tf.zeros(shape=[LABEL_COUNT]) + 0.1, name="sm_b")
                    tf.summary.histogram("output_weights", sm_w)
                    y = tensorflow_utils.activation_fun(args.activation, tf.nn.xw_plus_b(h_out, sm_w, sm_b))

            global_step = tf.Variable(0)

            loss = tensorflow_utils.loss_fun(args.loss, y, y_)
            tf.summary.scalar("loss", loss)

            train_op = tensorflow_utils.optimize_fun(args.gradient, args.learning_rate).minimize(loss, global_step=global_step)

            # Test trained model
            label = tf.argmax(y_, 1, name=constants.LABEL_NODE_NAME)
            prediction = tf.argmax(y, 1, name=constants.PREDICT_NODE_NAME)
            correct_prediction = tf.equal(prediction, label)

            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
            tf.summary.scalar("acc", accuracy)

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

        logdir = TFNode.hdfs_path(ctx, args.model_dir)
        print("tensorflow model path: {0}".format(logdir))
        summary_writer = tf.summary.FileWriter("tensorboard_%d" % (worker_num),
                                               graph=tf.get_default_graph())

        sv = tf.train.Supervisor(is_chief=(task_index == 0),
                                 logdir=logdir,
                                 init_op=init_op,
                                 summary_op=None,
                                 saver=saver,
                                 global_step=global_step,
                                 stop_grace_secs=300,
                                 save_model_secs=10)

        # The supervisor takes care of session initialization, restoring from a checkpoint, and closing when done or an error occurs.
        with sv.managed_session(server.target) as sess:
            print("{0} session ready".format(datetime.now().isoformat()))

            step = 0
            tf_feed = TFNode.DataFeed(ctx.mgr, input_mapping=args.input_mapping)
            while not sv.should_stop() and not tf_feed.should_stop() and step < args.steps:
            # while not sv.should_stop() and not tf_feed.should_stop():
                # Run a training step asynchronously.

                # using feed_dict
                batch_xs, batch_ys = feed_dict(tf_feed.next_batch(BATCH_SIZE), args)
                feed = {x: batch_xs, y_: batch_ys}

                if len(batch_xs) > 0:
                    _, summary, step = sess.run([train_op, summary_op, global_step],
                                                feed_dict=feed)
                    # print accuracy and save model checkpoint to HDFS every 100 steps
                    if (step % 100 == 0):
                        print("{0} step: {1} accuracy: {2}".format(datetime.now().isoformat(), step,
                                                                   sess.run(accuracy, {x: batch_xs, y_: batch_ys})))

                    if sv.is_chief:
                        summary_writer.add_summary(summary, step)

            if sv.should_stop() or step >= args.steps:
                tf_feed.terminate()

            # 保存模型
            if sv.is_chief and args.export_dir:
                print("{0} exporting saved_model to: {1}".format(datetime.now().isoformat(), args.export_dir))
                save_model(sess, args, x, prediction)

            else:
                # non-chief workers should wait for chief
                while not sv.should_stop():
                    print("Waiting for chief")
                    time.sleep(5)

        # Ask for all the services to stop.
        print("{0} stopping supervisor".format(datetime.now().isoformat()))
        sv.stop()


