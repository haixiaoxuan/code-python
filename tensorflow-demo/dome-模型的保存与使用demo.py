#!-*-coding=utf8-*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 使用 Saver 保存
def save_by_saver():
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    batch_size = 100
    batch_num = mnist.train.num_examples // batch_size

    x = tf.placeholder(tf.float32, (None, 28 * 28),name="x")
    y = tf.placeholder(tf.float32, (None, 10),name="y")

    weight = tf.Variable(tf.truncated_normal((28 * 28, 10)))
    bias = tf.Variable(tf.zeros((10)))
    y_predict = tf.nn.softmax(tf.nn.xw_plus_b(x, weight, bias))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_predict, y))
    train = tf.train.AdamOptimizer().minimize(loss)

    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_predict, axis=1),
                                          tf.argmax(y, axis=1)), tf.float32),name="accuracy")

    # 创建Saver对象
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2):
            for i in range(batch_num):
                x_data, y_data = mnist.train.next_batch(batch_size)
                sess.run(train, feed_dict={x: x_data, y: y_data})
            accuracy = sess.run(acc, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print(accuracy)
        # 模型保存
        saver.save(sess,"E:\\model\\test1-model")  # 前面为路径，最后为模型名

        # 模型加载
        # new_saver = tf.train.import_meta_graph('model/test-model.meta')
        # new_saver.restore(sess, tf.train.latest_checkpoint('model/'))
        #
        # accuracy = sess.run(acc, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        # print(accuracy)


# 通过name拿到参数
def get_by_name():
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('model/test-model.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('model/'))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        feed_dict={x: mnist.test.images, y: mnist.test.labels}

        accuracy = graph.get_tensor_by_name("accuracy:0")
        res = sess.run(accuracy,feed_dict=feed_dict)
        print(res)


# pb 格式模型保存
def save_by_pb():
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    batch_size = 100
    batch_num = mnist.train.num_examples // batch_size

    x = tf.placeholder(tf.float32, (None, 28 * 28), name="x")
    y = tf.placeholder(tf.float32, (None, 10), name="y")

    weight = tf.Variable(tf.truncated_normal((28 * 28, 10)))
    bias = tf.Variable(tf.zeros((10)))
    y_predict = tf.nn.softmax(tf.nn.xw_plus_b(x, weight, bias))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_predict, y))
    train = tf.train.AdamOptimizer().minimize(loss)

    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_predict, axis=1),
                                          tf.argmax(y, axis=1)), tf.float32), name="accuracy")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2):
            for i in range(batch_num):
                x_data, y_data = mnist.train.next_batch(batch_size)
                sess.run(train, feed_dict={x: x_data, y: y_data})
            accuracy = sess.run(acc, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print(accuracy)

        # 保存
        builder = tf.saved_model.builder.SavedModelBuilder("model/pb/")
        builder.add_meta_graph_and_variables(sess, ['tag_string'])
        builder.save()

        # 载入
        meta_graph_def = tf.saved_model.loader.load(sess, ['tag_string'], "model/pb/")
        x = sess.graph.get_tensor_by_name('x:0')
        y = sess.graph.get_tensor_by_name('y:0')
        # .............

# pb 格式模型保存(解耦)
def save_by_pb2():
    """ 那就需要给add_meta_graph_and_variables
    方法传入第三个参数，signature_def_map """
    x = tf.placeholder(tf.float32, (None, 28 * 28), name="x")
    y = tf.placeholder(tf.float32, (None, 10), name="y")

    with tf.Session() as sess:
        builder = tf.saved_model.builder.SavedModelBuilder("model/pb")
        # x 为输入tensor, keep_prob为dropout的prob tensor
        inputs = {'input_x': tf.saved_model.utils.build_tensor_info(x)} # 可以有多个

        # y 为最终需要的输出结果tensor
        outputs = {'output_y': tf.saved_model.utils.build_tensor_info(y)}

        signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, 'test_sig_name')

        builder.add_meta_graph_and_variables(sess, ['test_saved_model'], {'test_signature': signature})
        builder.save()


        # 加载
        signature_key = 'test_signature'
        input_key = 'input_x'
        output_key = 'output_y'

        meta_graph_def = tf.saved_model.loader.load(sess, ['test_saved_model'], "model/pb")
        # 从meta_graph_def中取出SignatureDef对象
        signature = meta_graph_def.signature_def

        # 如果不知道 signature_key，可以直接把signature打出来

        # 从signature中找出具体输入输出的tensor name
        x_tensor_name = signature[signature_key].inputs[input_key].name
        y_tensor_name = signature[signature_key].outputs[output_key].name

        # 获取tensor 并inference
        x = sess.graph.get_tensor_by_name(x_tensor_name)
        y = sess.graph.get_tensor_by_name(y_tensor_name)

        # _x 实际输入待inference的data
        sess.run(y,)  # ......


# 保存为整个pb文件
    def save_to_pb():
        """ 只需要保存要输出的 tensor 即可 """
        from tensorflow.python.framework import graph_util
        # constant_graph_input = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["input"])
        constant_graph_output = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output'])

        with tf.gfile.FastGFile('rf.pb', mode='wb') as f:
            # f.write(constant_graph_input.SerializeToString())
            f.write(constant_graph_output.SerializeToString())

        sess.close()

        # graph = tf.graph_util.convert_variables_to_constants(sess,
        #                                                      sess.graph_def,
        #                                                      ["output"])
        # tf.train.write_graph(graph, '.', 'rf.pb', as_text=False)


    def load_by_pb():
        """ 通过pb 模型进行预测 """
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            output_graph_def = tf.GraphDef()

            with open("model_path", "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")

            input = sess.graph.get_tensor_by_name("input:0")
            output = sess.graph.get_tensor_by_name("output:0")

            res = sess.run(output, feed_dict={input: "predict_data"})
            print(res)


if __name__=="__main__" :
    save_by_saver()
    # get_by_name()
    # save_by_pb()
