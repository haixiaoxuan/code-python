#！-*- encoding=utf8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.python.framework import graph_util
import pandas as pd


def one():
    mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

    # 定义批次大小 | 批次数
    batch_size = 100
    batch_number = mnist.train.num_examples // batch_size

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32,[None,10])

    # 设置初始 权重和偏度值
    weight = tf.Variable(tf.zeros([784,10]))
    bias = tf.Variable(tf.zeros([10]))

    # 使用softmax 作为 激活函数计算预测值
    predict = tf.nn.softmax(tf.matmul(x,weight)+bias)

    # 定义损失函数和 优化器
    # loss = tf.reduce_mean(tf.square(predict-y))
    # 优化，修改损失函数( 交叉熵代价函数 )
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y,predict))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict,y))
    train = tf.train.AdamOptimizer(0.1).minimize(loss)

    # 准确率计算
    arg_max = tf.argmax(predict,1)
    res = tf.equal(arg_max, tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(res, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(200):
            for j in range(batch_number):
                batch_x,batch_y = mnist.train.next_batch(batch_size)
                sess.run(train,feed_dict={x:batch_x,y:batch_y})
                # print(sess.run(res,feed_dict={x:batch_x,y:batch_y}))

            # 每迭代一次，计算一次准确率
            acc,index_max = sess.run([accuracy,arg_max], feed_dict={x: mnist.test.images,y: mnist.test.labels})
            print(i,acc)
            print(type(index_max))


def get_data(path):
    from sklearn import preprocessing
    onehot = preprocessing.OneHotEncoder()
    df = pd.read_csv(path)
    x = df.drop(["label"], axis=1).values
    y = df["label"].values

    # onehot.fit(y.reshape(-1, 1))
    # y = onehot.transform(y.reshape(-1, 1)).toarray()

    return x,y


def tf_model_test(train_x, train_y,
                  feature_num, label_num, h_info,
                  batch_size, learn_rate,
                  optimize_name, epoches):

    train_y = tf.one_hot(train_y, 10)

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
    def generate_hidden_layer(in_x, in_count, out_count, name, seed=1):
        weight = tf.Variable(tf.truncated_normal([in_count, out_count], stddev=0.1, seed=seed),
                             name="hidden_weight_" + name)
        biase = tf.Variable(tf.zeros([out_count]) + 0.1, name="hidden_biase_" + name)
        return tf.nn.xw_plus_b(in_x, weight, biase)

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

    def optimize_fun(gradient_option, learning_rate):
        """ 选择优化器，默认梯度下降 """
        if gradient_option == "GradientDescent":  # 包括：随机梯度SGD, 批梯度BGD, Mini批梯度MBGD
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif gradient_option == "Momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=1)
        elif gradient_option == "Adagrad":
            optimizer = tf.train.AdagradOptimizer(learning_rate)
        elif gradient_option == "Adam":
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif gradient_option == "Adadelta":
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        elif gradient_option == "Ftrl":
            optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
        elif gradient_option == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        return optimizer

    # 定义批次大小 | 批次数
    batch_number = train_x.shape[0] // batch_size

    x = tf.placeholder(tf.float32, [None, feature_num], name="input")
    y = tf.placeholder(tf.float32, [None, label_num])

    layers_list = []  # 定义隐藏层每一层神经元的个数和输出
    layers_list.append([feature_num, x])
    hidden_info = get_hidden_layer_info(h_info)
    if hidden_info != None and len(hidden_info) != 0:
        with tf.name_scope('hidden_layer'):
            for index, lay in enumerate(hidden_info):
                h_out = generate_hidden_layer(layers_list[index][1],
                                              layers_list[index][0], int(lay[0]), str(index), 1)
                out = activation_fun(lay[1], h_out)
                layers_list.append([int(lay[0]), out])

    # 输出层
    out_info = layers_list.pop()
    weight_l2 = tf.Variable(tf.truncated_normal([out_info[0], label_num], stddev=0.1))
    bias_l2 = tf.Variable(tf.zeros([label_num]) + 0.1)
    # 使用softmax 作为 激活函数计算预测值
    predict = tf.nn.softmax(tf.matmul(out_info[1], weight_l2) + bias_l2)

    # 定义损失函数和 优化器
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predict))
    train = optimize_fun(optimize_name, learn_rate).minimize(loss)

    # 准确率计算
    arg_max = tf.argmax(predict, axis=1, name="output")
    res = tf.equal(arg_max, tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(res, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_y = sess.run(train_y)

        for i in range(epoches):

            for j in range(batch_number):
                batch_x = train_x[j:(j + 1) * batch_size]
                batch_y = train_y[j:(j + 1) * batch_size]
                sess.run(train, feed_dict={x: batch_x, y: batch_y})

            # 每迭代一次，计算一次准确率
            acc, max_index = sess.run([accuracy, arg_max], feed_dict={x: train_x, y: train_y})
            print(str(i) + "steps  accuracy:" + str(acc))


        # 保存模型( 保存为单个pb )
        constant_graph_output = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output'])
        with tf.gfile.FastGFile('model.pb', mode='wb') as f:
            f.write(constant_graph_output.SerializeToString())

        # 保存为( 图参数分离的pb )
        builder = tf.saved_model.builder.SavedModelBuilder("pb")
        inputs = {'input_x': tf.saved_model.utils.build_tensor_info(x)} # 可以有多个
        outputs = {'output_y': tf.saved_model.utils.build_tensor_info(arg_max)}

        signature = tf.saved_model.signature_def_utils\
            .build_signature_def(inputs, outputs, 'test_sig_name')

        builder.add_meta_graph_and_variables(sess,
                                             ['test_saved_model'],
                                             {'test_signature': signature})
        builder.save()



def predict_by_pb(model_path, predict_data):
    """ 通过pb 模型进行预测 """
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        output_graph_def = tf.GraphDef()

        with open(model_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def,name="")

        input = sess.graph.get_tensor_by_name("input:0")
        output = sess.graph.get_tensor_by_name("output:0")

        res = sess.run(output, feed_dict={input: predict_data})
        print(res)

def predict_by_pb2(model_path, predict_data):
    with tf.Session() as sess:
        signature_key = 'test_signature'
        input_key = 'input_x'
        output_key = 'output_y'

        meta_graph_def = tf.saved_model.loader.load(sess, ['test_saved_model'], model_path)
        # 从meta_graph_def中取出SignatureDef对象
        signature = meta_graph_def.signature_def

        # 从signature中找出具体输入输出的tensor name
        x_tensor_name = signature[signature_key].inputs[input_key].name
        y_tensor_name = signature[signature_key].outputs[output_key].name

        # 获取tensor 并inference
        x = sess.graph.get_tensor_by_name(x_tensor_name)
        y = sess.graph.get_tensor_by_name(y_tensor_name)

        res = sess.run(y, feed_dict={x: predict_data})
        print(res)


if __name__=="__main__":
    path = "E:\\data\\stand_data.csv"
    x, y = get_data(path)
    #
    # # 训练
    tf_model_test(x, y,
                  feature_num=784,
                  label_num=10,
                  h_info="1000,Tanh|1000,Tanh",
                  batch_size=128,
                  learn_rate=0.1,
                  optimize_name="GradientDescent",
                  epoches=1)

    # 使用 单个 pb 预测
    # predict_x = pd.read_csv("E:/data/data_cell_test_stand.csv").values
    # predict_by_pb("model.pb", predict_x)

    # 使用 分离pb 预测
    # predict_by_pb2("pb", predict_x)



















