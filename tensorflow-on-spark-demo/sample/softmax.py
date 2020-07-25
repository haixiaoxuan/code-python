#!-*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import HiveContext, SQLContext
from tensorflowonspark import TFCluster
import numpy as np

import softmax_dist


# 设置参数解析器
def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app_name", help="the name of spark app", default="spark_app")
    parser.add_argument("--feature_count", help="the count of feature field", type=int, default=1)
    parser.add_argument("--batch_size", help="number of records per batch", type=int, default=100)
    parser.add_argument("--learning_rate", help="the learning rate for training", type=float, default=0.01)
    parser.add_argument("--activation", help="ReLU|ReLU6|Sigmoid|Tanh", default="Sigmoid")      # 目前定死的 最后一层 softmax
    parser.add_argument("--loss", help="L2|L1|SigmoidCrossEntropy", default="SigmoidCrossEntropy")
    parser.add_argument("--gradient", help="GradientDescent|Adam|Adadelta|Ftrl|RMSProp|Momentum|Adagrad",
                        default="GradientDescent")
    parser.add_argument("--epochs", help="number of epochs", type=int, default=1)
    parser.add_argument("--export_dir", help="HDFS path to export saved_model", default="export")  # pb 模型文件保存路径
    parser.add_argument("--format", help="example format: (csv|pickle|tfr)", choices=["csv", "pickle", "tfr"],
                        default="csv")
    parser.add_argument("--train_path", help="HDFS path to train data path")  # 样本feature的路径

    parser.add_argument("--model", help="HDFS path to save/load model during train/inference",
                        default="model")  # mode 保存路径
    parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int)
    parser.add_argument("--prediction", help="HDFS path to save test/inference output",
                        default="predictions")  # predict 输出路径
    parser.add_argument("--evaluation", help="HDFS path to save evaluation result", default="evaluation")
    parser.add_argument("--steps", help="maximum number of steps", type=int, default=1000)
    parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")
    parser.add_argument("--mode", help="train|inference", default="train")
    parser.add_argument("--rdma", help="use rdma connection", default=False)

    parser.add_argument("--layers_active", help="隐层层的层数、和每层的神经元个数、以及每层的激活函数")
    parser.add_argument("--seed", help="seed", type=int ,default=0)

    parser.add_argument("--label_name", help="标签列的名称")
    parser.add_argument("--sample_ratio", help="样本占比",type=float ,default=0.3)
    parser.add_argument("--label_count", help="the count of label field", type=int, default=1)

    return parser


# 第一列为label, 第二列为预测值
def model_evaluate(sc, prediction_rdd, evaluation_save_path):
    """evaluating the model"""

    list_rdd = prediction_rdd.map(lambda lines: 1 if lines.split(",")[0] == lines.split(",")[1] else 0)

    accuracy = list_rdd.mean()

    result_string = "Accuracy: {0}".format(accuracy)
    print(result_string)
    save_evaluation_to_file(sc, evaluation_save_path, result_string)


# 保存模型评测结果
def save_evaluation_to_file(sc, save_path, eval_result):
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    Configuration = sc._gateway.jvm.org.apache.hadoop.conf.Configuration
    fs = FileSystem.get(Configuration())
    output = fs.create(Path(save_path + '/model.evtxt'))

    output.write(bytearray(eval_result,encoding="utf8"))
    output.close()

# 读取hdfs数据
def read_train_data(hiveContext,label_name,args,feature_alias):
    data_df = hiveContext.read.csv(args.train_path, header=True, inferSchema=True)
    all_fields = data_df.columns
    all_fields.remove(label_name)
    features_name =[feature_name for index,feature_name in enumerate(all_fields) if index < args.feature_count]
    # 注册临时表对特征和标签进行区分
    tmp_table_name = "tmp_train_table"
    data_df.registerTempTable(tmp_table_name)
    feature_name_list = ["`" + i + "`" for i in features_name]
    sql = "select concat({0}) as {1},{2} from {3}".format(",',',".join(feature_name_list), feature_alias, label_name,
                                                          tmp_table_name)
    train_df = hiveContext.sql(sql)
    return train_df

# 对标签进行编码，并对 feature进行切分
def label_one_hot(row):
    one_hot_label = np.zeros(type_count)
    label_index = label_type.index(row.asDict()[label_name])
    one_hot_label[label_index] = 1
    return ([float(i) for i in row.asDict()[feature_alias].split(",")], one_hot_label.tolist())

def extract_label_species(train_df,label_name):
    # 提取类别数
    label_type = [i.asDict()[label_name] for i in train_df.select(label_name).distinct().collect()]
    type_count = len(label_type)
    return label_type,type_count


# 解析命令行参数
parser = create_arg_parser()
args = parser.parse_args()

# 取得Spark配置中application运行配置的executors的数量
sc = SparkContext(conf=SparkConf().setAppName(args.app_name))
sc.setLogLevel("WARN")
hiveContext = HiveContext(sc)
executors = sc._conf.get("spark.executor.instances")
num_executors = int(executors) if executors is not None else 1

# 指定 parameter server的个数
num_ps = 1

# 如果命令行未指定tensorflow集群的大小，则采用spark配置中指定的executors数量作为集群的大小
if args.cluster_size == None:
    args.cluster_size = num_executors

print("args:", args)
print("{0} ===== Start".format(datetime.now().isoformat()))

label_name = args.label_name

feature_alias = "feature"    # 特征字段的别名
# 读取训练数据并提取特征字段名
train_df = read_train_data(hiveContext, label_name, args , feature_alias)

# 提取特征类别数
label_type , type_count = extract_label_species(train_df,label_name)
args.label_count = type_count

# 进行 one-hot 编码
dataRDD = train_df.rdd.map(label_one_hot)

trainRDD , testRDD = dataRDD.randomSplit([1 - float(args.sample_ratio), float(args.sample_ratio)],seed=args.seed)

# 构建tensorflow on spark集群
cluster = TFCluster.run(sc,
                        softmax_dist.map_fun,
                        args,
                        args.cluster_size,  # 集群节点个数
                        num_ps,
                        args.tensorboard,
                        TFCluster.InputMode.SPARK,
                        log_dir=args.model)
print("{0} ===== Train Start".format(datetime.now().isoformat()))
# 模型训练
cluster.train(trainRDD, args.epochs)
# 关闭集群
cluster.shutdown(grace_secs=30)  # 采用graceful方式关闭tensorflow on spark集群
print("{0} ===== Train Stop".format(datetime.now().isoformat()))

# 构建tensorflow on spark集群
args.mode = "inference"
cluster = TFCluster.run(sc,
                        softmax_dist.map_fun,
                        args,
                        args.cluster_size,  # 集群节点个数
                        num_ps,
                        args.tensorboard,
                        TFCluster.InputMode.SPARK,
                        log_dir=args.model)
print("{0} ===== Inference Start".format(datetime.now().isoformat()))
labelRDD = cluster.inference(testRDD)
print("===== Evaludating model")
model_evaluate(sc, labelRDD, args.evaluation)  # 模型评估并保存结果
labelRDD.saveAsTextFile(args.prediction)  # 保存预测结果

cluster.shutdown(grace_secs=30)  # 采用graceful方式关闭tensorflow on spark集群

print("{0} ===== Inference Stop".format(datetime.now().isoformat()))



