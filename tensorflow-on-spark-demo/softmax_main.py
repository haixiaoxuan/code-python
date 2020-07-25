#! -*-coding=utf8-*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from pyspark import Row
from pyspark.sql import SparkSession
from tensorflowonspark.pipeline import TFEstimator

import argparse
from datetime import datetime

from algorithm_utils import features_processing
from algorithm_utils import common_utils
from algorithm_utils import load_data
from algorithm_utils import save_data
from algorithm_utils import softmax_dist
from algorithm_utils import index_calculation
from algorithm_utils import constants


def parse_args():
    parser = argparse.ArgumentParser()

    # 路径参数
    parser.add_argument("--train_path", help="HDFS path to MNIST images in parallelized format", type=str)
    parser.add_argument("--output", help="HDFS path to save test/inference output", type=str)
    parser.add_argument("--evaluate", help="HDFS path to save test/inference evaluate result", type=str)  # 存放结果
    parser.add_argument("--model_dir", help="HDFS path to save/load model during train/inference", type=str)
    parser.add_argument("--export_dir", help="HDFS path to export pd model", type=str)

    # 算法参数
    parser.add_argument("--batch_size", help="number of records per batch", type=int, default=100)
    parser.add_argument("--epochs", help="number of epochs", type=int, default=1)
    parser.add_argument("--steps", help="maximum number of steps", type=int, default=2000)  # 已经弃用
    parser.add_argument("--learning_rate", help="the learning rate for training", type=float, default=0.01)
    parser.add_argument("--sample_ratio", help="样本占比", type=float, default=0.3)
    parser.add_argument("--layers_active", help="隐层层的层数、和每层的神经元个数、以及每层的激活函数")
    parser.add_argument("--label_count", help="the count of label field", type=int, default=1)
    parser.add_argument("--seed", help="seed", type=int, default=0)
    parser.add_argument("--label_name", help="标签列的名称")
    parser.add_argument("--feature_count", help="the count of feature field", type=int, default=1)
    parser.add_argument("--loss", help="L2|L1|SigmoidCrossEntropy|SoftmaxCrossEntropy", default="SoftmaxCrossEntropy")
    parser.add_argument("--activation", help="ReLU|ReLU6|Sigmoid|Tanh|Softmax", default="Sigmoid")  # 目前定死的 最后一层 softmax
    parser.add_argument("--gradient", help="GradientDescent|Adam|Adadelta|Ftrl|RMSProp|Momentum|Adagrad",
                        default="GradientDescent")

    # cluster conf
    parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int)
    parser.add_argument("--app_name", help="the name of spark app", default="spark_app")
    parser.add_argument("--num_ps", help="number of PS nodes in cluster", type=int, default=1)

    # 传递变量的默认参数
    parser.add_argument("--feature_alias", type=str, default="feature")

    parser.add_argument("--protocol", help="Tensorflow network protocol (grpc|rdma)", default="grpc")
    parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")

    args = parser.parse_args()
    print("args:", args)
    return args


def init_spark_session(args):
    """
    初始化 sparksession
    :param args:
    :return: sc sparkContext
             spark sparkSession
             num_executor executor的数量
    """
    spark = SparkSession \
        .builder \
        .appName(args.app_name) \
        .enableHiveSupport() \
        .getOrCreate()
    sc = spark.sparkContext

    # 获取executor的数量
    executors = sc._conf.get("spark.executor.instances")
    num_executors = int(executors) if executors is not None else 1

    # 设置 rdd 和sparksql 的并行度
    sc.setLogLevel("WARN")
    sc.getConf().set("spark.defalut.parallelism", num_executors)
    spark.conf.set("spark.sql.shuffle.partitions", num_executors)

    # 设置 tensorflow 集群大小
    args.cluster_size = num_executors if args.cluster_size == None else args.cluster_size
    print("{0} ===== Start".format(datetime.now().isoformat()))
    return sc, spark, num_executors


def get_tf_estimator(args):
    """ 返回一个 tensorflow estimator """
    estimator = TFEstimator(softmax_dist.map_fun, args) \
        .setInputMapping({args.feature_alias: args.feature_alias,
                          args.label_name: args.label_name}) \
        .setModelDir(args.model_dir) \
        .setExportDir(args.export_dir) \
        .setClusterSize(args.cluster_size) \
        .setNumPS(args.num_ps) \
        .setProtocol(args.protocol) \
        .setTensorboard(args.tensorboard) \
        .setEpochs(args.epochs) \
        .setBatchSize(args.batch_size) \
        .setSteps(args.steps)
    return estimator


def set_model_param(model, args):
    model.setModelDir(args.model_dir)
    model.setExportDir(None)
    model.setInputMapping({args.feature_alias: constants.INPUT_LAYER_NAME + "/" + constants.INPUT_LAYER_X,
                           args.label_name: constants.INPUT_LAYER_NAME + "/" + constants.INPUT_LAYER_Y})  # column -> tensor
    model.setOutputMapping({constants.INPUT_LAYER_NAME + "/" + constants.INPUT_LAYER_X: "features",
                            constants.LABEL_NODE_NAME: "label",
                            constants.PREDICT_NODE_NAME: "prediction"})  # tensor -> column


def get_train_data(spark, args, num_executors):
    """ 拿到经过 etl 之后的训练数据 """
    # 读取训练数据并提取特征字段名 [feature,label]
    input_dict = eval(args.train_path)
    assert type(input_dict) == dict, "train_path should is json but %s" % \
                                     str(type(input_dict))

    data_df = load_data.get_data(spark, input_dict, num_executors)
    all_fields = data_df.columns
    all_fields.remove(args.label_name)

    # 特征字段列表
    features_name = [feature_name for index, feature_name in enumerate(all_fields)
                     if index < args.feature_count]
    args.feature_count = len(features_name)

    # 过滤空行
    data_df = data_df.dropna().cache()

    # 对label进行one-hot编码并进行transform -> [[features],[label]]
    data_rdd, label_type, type_count = features_processing \
        .agg_features_and_one_hot_label(data_df,
                                        features_name, args.label_name)

    args.label_count = type_count
    data_df.unpersist()
    dataDF = spark.createDataFrame(data_rdd, [args.feature_alias, args.label_name])

    return dataDF, label_type, features_name


def save_res_to_hdfs(test_preds_tmp, args, label_type, features_name, spark):
    """ 将预测结果保存到 HDFS """
    test_preds_tmp = test_preds_tmp.rdd.map(lambda row: [row.asDict()[i]
                                                         for i in ["features", "label", "prediction"]]) \
        .cache()

    # 将one-hot编码之后的预测结果进行翻译，并保存至 HDFS
    features_name.append(args.label_name).append("predict")
    sample_row = Row(*features_name)

    def _generate_row(row):
        row_feature = row[0]
        row_feature.append(row[1]).append(row[2])
        return sample_row(*row_feature)

    test_preds_res = test_preds_tmp.map(_generate_row)
    spark.createDataFrame(test_preds_res).write.csv(args.output, header=True)

    return test_preds_tmp


def evaluate_model(test_preds, label_type):
    # 总准确率计算
    total_predict = test_preds.map(lambda row: 1 if row[0] == row[1] else 0).mean()
    print("总准确率：" + str(total_predict))

    # 每个分类的精准率和召回率计算
    final_result_tmp = index_calculation.cal_recall_and_precise(test_preds)

    # 将模型评测结果中的类别编码翻译为具体类别
    final_result = {}
    for line in final_result_tmp:
        for k, v in line.items():
            key = label_type[k]
            final_result[key] = v

    for label_t in label_type:
        if final_result.get(label_t) == None:
            final_result[label_t] = {"recall": 0, "precise": 0}

    # 将评测结果写入文件
    other_indicators = []
    for key, value in final_result.items():
        value["type"] = key
        other_indicators.append(value)
    final_result_dic = {"total_accuracy": total_predict, "other_indicators": other_indicators}
    print(final_result_dic)

    return final_result


def run():
    args = parse_args()
    sc, spark, num_executors = init_spark_session(args)

    # 拿到训练数据
    dataDF, label_type, features_name = get_train_data(spark, args, num_executors)

    trainDF, testDF = dataDF.randomSplit([1 - float(args.sample_ratio),
                                          float(args.sample_ratio)],
                                         seed=args.seed)

    trainDF.cache()
    testDF.cache()  # 后面用于预测

    import math
    trainDFCount = trainDF.count()
    args.epochs = math.ceil(args.steps * args.batch_size / trainDFCount)
    print("迭代步数：" + str(args.steps) + "=======" + " 迭代轮次：" + str(args.epochs))

    # 将路径转换为 hdfs 路径
    args.export_dir = common_utils.get_abs_path(sc, args.export_dir)
    args.model_dir = common_utils.get_abs_path(sc, args.model_dir)

    # 训练
    print("{0} ===== Estimator.fit()".format(datetime.now().isoformat()))
    estimator = get_tf_estimator(args)
    model = estimator.fit(trainDF)
    trainDF.unpersist()

    # 推理 [y_,y_pre]
    set_model_param(model, args)
    print("{0} ===== Model.transform()".format(datetime.now().isoformat()))
    test_preds_tmp = model.transform(testDF)
    testDF.unpersist()

    test_preds_tmp = save_res_to_hdfs(test_preds_tmp, args,
                                      label_type, features_name, spark)

    # 将预测结果和label拿出来，评估模型
    test_preds = test_preds_tmp.map(lambda row: [row[1], row[2]]).cache()
    test_preds_tmp.unpersist()

    # 模型评估
    final_result = evaluate_model(test_preds, label_type)
    save_data.write_data_to_cluster(sc,
                                    args.evaluate + constants.PATH_SEP + constants.EVALUATION_MODEL_NAME,
                                    str(final_result))

    print("{0} ===== Stop".format(datetime.now().isoformat()))


if __name__ == "__main__":
    run()
