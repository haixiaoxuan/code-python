
from pyspark.sql import SparkSession
from spark_sklearn import Converter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn2pmml import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from sklearn.externals import joblib
import pandas as pd

import argparse
import time
import os

from algorithm_utils import load_data
from algorithm_utils import constants
from algorithm_utils import save_data
from algorithm_utils import drawing_pic


def parse_args():
    """ 命令行参数解析 """
    parser = argparse.ArgumentParser()

    # 路径参数
    parser.add_argument("--train_path", help="HDFS path to train data", type=str)
    parser.add_argument("--output", help="HDFS path to save test/inference output", type=str)
    parser.add_argument("--evaluate", help="HDFS path to save test/inference evaluate result", type=str)  # 存放结果
    parser.add_argument("--model_dir", help="HDFS path to save/load model during train/inference", type=str)
    parser.add_argument("--export_dir", help="HDFS path to export pmml model", type=str)

    # 算法参数
    parser.add_argument("--n_estimators", help="弱学习器个数", type=int, default=100)
    parser.add_argument("--learning_rate", help="每个弱学习器的权重系数", type=float, default=0.01)
    parser.add_argument("--subsample", help="不放回采样", type=int, default=1)
    parser.add_argument("--loss", help="损失函数", type=str, default="deviance")
    parser.add_argument("--max_features", help="最大特征数 auto|sqrt|log2|None|数字", type=str, default="None")
    parser.add_argument("--max_depth", help="数的最大深度 数字|None", type=str, default=None)
    parser.add_argument("--min_sample_split", help="节点分裂的最小样本数", type=int, default=2)
    parser.add_argument("--min_sample_leaf", help="叶节点的最小样本数，小于则剪枝", type=int, default=1)
    parser.add_argument("--min_weight_fraction_leaf", help="叶子节点所有样本权重和的最小值，小于则剪枝", type=int, default=0)
    parser.add_argument("--max_leaf_nodes", help="最大叶子节点数 int|None", type=str, default=None)
    parser.add_argument("--min_impurity_decrease", help="最小节点分裂不纯度减少量", type=float, default=0)
    parser.add_argument("--presort", help="是否启用预排序 boolean|auto", type=str, default="auto")

    parser.add_argument("--sample_ratio", help="测试样本占比", type=float, default=0.3)
    parser.add_argument("--seed", help="seed", type=int, default=0)
    parser.add_argument("--label_count", help="类别数", type=int, default=1)
    parser.add_argument("--label_name", help="标签列的名称")
    parser.add_argument("--feature_count", help="the count of feature field", type=int, default=1)

    # cluster conf
    parser.add_argument("--app_name", help="the name of spark app", default="GBDT")

    args = parser.parse_args()
    print("========> args:", args)
    return args


def get_model(args):
    """ 根据参数获取 gbdt 模型 """

    # 对没有固定类型的参数进行单独处理
    max_features = args.max_features if args.max_features in ["auto", "sqrt", "log2"] \
                                    else eval(args.max_features)
    max_depth = eval(args.max_depth)
    max_leaf_nodes = eval(args.max_leaf_nodes)
    presort = "auto" if args.presort.lower() == "auto" else eval(args.presort)

    model = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        loss=args.loss,
        max_features=max_features,
        max_depth=max_depth,
        min_samples_split=args.min_sample_split,
        min_samples_leaf=args.min_sample_leaf,
        min_weight_fraction_leaf=args.min_weight_fraction_leaf,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=args.min_impurity_decrease,
        presort=presort,
        verbose=1
    )
    print("=======> 开始训练")
    return model


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
    sc.getConf().set("spark.defalut.parallelism", num_executors * 4)
    spark.conf.set("spark.sql.shuffle.partitions", num_executors * 4)

    return spark, int(executors), sc


def get_train_data(args, spark, num_executors):
    """ 得到训练数据 """
    input_dict = eval(args.train_path)
    assert type(input_dict) == dict, "train_path should is json but %s" % \
                                     str(type(input_dict))
    data_df = load_data.get_data(spark, input_dict, num_executors*4)
    return data_df


def sparkDF2pandasDF(sc, df):
    """ 将 spark DataFrame 转换为 pandas DataFrame """
    converter = Converter(sc)
    return converter.toPandas(df)


def gen_pmml_to_hdfs(sc, model, args):
    """ 生成 pmml,pkl 到 hdfs """

    # 先将pmml文件生成到driver的tmp目录下，然后再将其上传至HDFS
    # 目录名称 /tmp/时间戳/文件名
    print("===========> 保存模型文件到 HDFS")
    pmml_model_name = constants.PMML_NAME
    pkl_model_name = constants.PKL_NAME
    dir_name = "/tmp/" + str(time.time())
    args.tmp_dir = dir_name
    os.mkdir(dir_name)

    # 保存为pmml文件
    pipeline = PMMLPipeline([("classifier", model)])
    sklearn2pmml(pipeline, dir_name + os.sep + pmml_model_name, with_repr=True)
    joblib.dump(model, dir_name + os.sep + pkl_model_name)

    # 上传文件至HDFS
    with open(dir_name + os.sep + pmml_model_name, "r") as f1, \
            open(dir_name + os.sep + pkl_model_name, "rb") as f2:
        data1 = f1.read()
        data2 = f2.read()
        save_data.write_data_to_cluster(sc, args.export_dir + os.sep + pmml_model_name, data1)
        save_data.write_data_to_cluster(sc, args.model_dir + os.sep + pkl_model_name, data2, is_text_file=False)

    # 删除临时文件
    os.remove(dir_name + os.sep + pmml_model_name)
    os.remove(dir_name + os.sep + pkl_model_name)


def evaluate_model(y, y_, args, class_s, sc, model, x):
    """ 评估模型 """
    print("=======> 开始进行模型评估")

    # 保存报告文件到HDFS
    report = classification_report(y, y_, labels=class_s)
    report_dict = classification_report(y, y_, labels=class_s, output_dict=True)
    res_dict = {}
    accuracy_score(y, y_)
    res_dict["total_accuracy"] = accuracy_score(y, y_)

    # 存放类别字典
    class_list = []
    res_dict["other_indicators"] = class_list
    for i in [str(c) for c in class_s]:
        d = {"recall": report_dict[i]["recall"],
             "precise": report_dict[i]["precision"],
             "f1-score": report_dict[i]["f1-score"],
             "type": i
             }
        class_list.append(d)
    res_dict["n_class"] = len(class_s)

    save_data.write_data_to_cluster(sc, args.evaluate + os.sep + constants.CLASSIFICATION_REPORT, report)

    # 保存报告图片
    report_path = args.tmp_dir + os.sep + constants.CLASSIFICATION_REPORT + ".png"
    drawing_pic.save_text_to_pic(report, report_path)
    with open(report_path, "rb") as f:
        data = f.read()
        save_data.write_data_to_cluster(sc,
                                        args.evaluate + os.sep + constants.CLASSIFICATION_REPORT + ".png",
                                        data,
                                        is_text_file=False)
    os.remove(report_path)

    # 保存混淆矩阵信息到HDFS
    matrix = confusion_matrix(y, y_, labels=class_s).tolist()
    res = ""
    for index, line in enumerate(matrix):
        res += str(class_s[index]) + "," + ",".join([str(l) for l in line]) + "\n"
    save_data.write_data_to_cluster(sc, args.evaluate + os.sep + constants.CONFUSION_MATRIX, res)

    # 保存混淆矩阵图片到HDFS
    matrix_path = args.tmp_dir + os.sep + constants.CONFUSION_MATRIX + ".png"
    drawing_pic.save_confusion_matrix(matrix, class_s, matrix_path)
    with open(matrix_path, "rb") as f:
        data = f.read()
        save_data.write_data_to_cluster(sc,
                                        args.evaluate + os.sep + constants.CONFUSION_MATRIX + ".png",
                                        data,
                                        is_text_file=False)
    os.remove(matrix_path)

    if len(class_s) == 2 :
        # 如果是二分类，则需要生成 ROC曲线

        # 计算出预测概率
        y_proba = model.predict_proba(x)

        # 取第一个类为 正样例
        fpr, tpr, thresholds = roc_curve(y, y_proba[:, 0], pos_label=class_s[0])
        roc_auc = auc(fpr, tpr)
        roc_pic_path = args.tmp_dir + os.sep + constants.ROC_PIC
        drawing_pic.save_roc_to_pic(fpr, tpr, roc_auc, roc_pic_path)
        with open(roc_pic_path, "rb") as f:
            data = f.read()
            save_data.write_data_to_cluster(sc,
                                            args.evaluate + os.sep + constants.ROC_PIC,
                                            data,
                                            is_text_file=False)
        os.remove(roc_pic_path)
        res_dict["auc"] = roc_auc

    # 删除临时目录
    os.rmdir(args.tmp_dir)
    # 将评估结果写入HDFS
    save_data.write_data_to_cluster(sc, args.evaluate + os.sep + constants.EVALUATION_MODEL_NAME, str(res_dict))



def main():

    args = parse_args()
    spark, executors, sc = init_spark_session(args)
    data_df = get_train_data(args, spark, executors)
    # 删除空值
    data_df = data_df.dropna()

    # 转换为 pandas dataframe
    data_df = sparkDF2pandasDF(sc, data_df)
    x = data_df.drop([args.label_name], axis=1)
    y = data_df[args.label_name]

    # 随机切分进行交叉验证
    train_x, test_x, train_y, test_y = train_test_split \
        (x, y, test_size=args.sample_ratio, random_state=args.seed)

    # 初始化GBDT模型
    model = get_model(args)
    model.fit(train_x, train_y)

    # 保存模型， 包括两种模型(pmml|pkl)
    gen_pmml_to_hdfs(sc, model, args)

    # 预测
    test_y_ = model.predict(test_x)

    # 将预测结果和原始结果进行合并之后将结果输出到HDFS
    print("=========> 保存预测结果到HDFS")
    test_y_ = pd.DataFrame(test_y_, index=test_y.index, columns=["predict"])
    df = pd.concat((test_x, test_y, test_y_), axis=1)
    spark.createDataFrame(df).write.csv(args.output, header=True)

    # 模型评估
    evaluate_model(test_y, test_y_, args, model.classes_, sc, model, test_x)
    print("========> over")


if __name__ == "__main__":
    main()