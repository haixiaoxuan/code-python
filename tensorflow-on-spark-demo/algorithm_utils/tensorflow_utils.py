#!-*-coding=utf8-*-

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.saved_model import tag_constants

def activation_fun(activation_option, input_maxtrix):
    """ 激活函数,默认 relu """
    if activation_option == "ReLU":
        hid = tf.nn.relu(input_maxtrix)
    elif activation_option == "ReLU6":
        hid = tf.nn.relu6(input_maxtrix)
    elif activation_option == "Tanh":
        hid = tf.nn.tanh(input_maxtrix)
    elif activation_option == "Sigmoid":
        hid = tf.nn.sigmoid(input_maxtrix)
    elif activation_option == "Softmax":
        hid = tf.nn.softmax(input_maxtrix)
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


def loss_fun(loss_option, output, label):
    """ 选择损失函数，默认sigmoid交叉熵 """
    if loss_option == "SigmoidCrossEntropy":
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=label))
    elif loss_option == "L2":
        loss = tf.nn.l2_loss(label - output)
    elif loss_option == "L1":
        loss = tf.reduce_mean(tf.abs(label - output))
    elif loss_option == "SoftmaxCrossEntropy":
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=label))
        tf.nn.cross
    else:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=label))

    return loss

def get_hidden_layer_info(arg):
    """ 解析隐藏层参数 """
    res = []
    if arg != None and len(arg) != 0:
        for i in arg.split("|"):
            tmp = []
            for j in i.split(","):
                tmp.append(j)
            res.append(tmp)
    return res


def hidden_layer_fun(in_x, in_count, out_count, name, seed):
    """ 隐藏层函数 """
    weight = tf.Variable(tf.truncated_normal([in_count, out_count], seed=seed, stddev=0.1),
                         name="hidden_weight_" + name)
    biase = tf.Variable(tf.zeros([out_count]) + 0.1, name="hidden_biase_" + name)
    return tf.nn.xw_plus_b(in_x, weight, biase)


def auto_generate_hidden_layer(n_count, out, layers_active, seed):
    """ 自动生成隐藏层 """
    layers_list = []  # 定义隐藏层每一层神经元的个数和输出
    layers_list.append([n_count, out])
    hidden_info = get_hidden_layer_info(layers_active)
    if hidden_info != None and len(hidden_info) != 0:
        with tf.name_scope("hidden_layer"):
            for index, lay in enumerate(hidden_info):
                h_out = hidden_layer_fun(layers_list[index][1], layers_list[index][0], int(lay[0]),
                                         str(index), seed=seed)
                out = activation_fun(lay[1], h_out)
                layers_list.append([int(lay[0]), out])
    return layers_list.pop()


def convert_as_single_pb(input_saved_model_dir, output_node_names, output_graph):
    """ 将图参数分离的pb文件转为单个pb文件 """
    freeze_graph.freeze_graph(
        input_saved_model_dir=input_saved_model_dir,
        output_node_names=output_node_names,
        output_graph=output_graph,
        input_graph="",
        input_saver="",
        input_binary=False,
        input_checkpoint="",
        restore_op_name="save/restore_all",
        filename_tensor_name="save/Const:0",
        clear_devices=True,
        initializer_nodes="",
        variable_names_whitelist="",
        variable_names_blacklist="",
        input_meta_graph="",
        saved_model_tags=tag_constants.SERVING,
        checkpoint_version=saver_pb2.SaverDef.V2)
    print("========= save single pb successful")
