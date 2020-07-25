import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
from pyspark.sql import SparkSession
from spark_sklearn import Converter
from algorithm_utils import constants
from algorithm_utils import save_data
from algorithm_utils import load_data
from algorithm_utils import drawing_pic
import argparse
import math
import copy
import time
import os
from multiprocessing import Pool

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


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
    parser.add_argument("--growth", help="用来规定线性或逻辑曲线趋势", type=str, default="linear")
    parser.add_argument("--changepoint_range", help="估计趋势变化点的历史比例", type=float, default=0.8)
    parser.add_argument("--yearly_seasonality", help="分析数据的年季节性", type=str, default="auto")
    parser.add_argument("--weekly_seasonality", help="分析数据的周季节性", type=str, default="auto")
    parser.add_argument("--daily_seasonality", help="分析数据的天季节性", type=str, default="auto")
    parser.add_argument("--seasonality_mode", help="季节模型", type=str, default="additive")
    parser.add_argument("--seasonality_prior_scale", help="季节性组件的强度", type=float, default=10)
    parser.add_argument("--holidays_prior_scale", help="节假日模型组件的强度", type=float, default=10)
    parser.add_argument("--changepoint_prior_scale", help="增长趋势模型的灵活度", type=float, default=0.05)
    parser.add_argument("--mcmc_samples", help="mcmc采样，用于获得预测未来的不确定性", type=int, default=0)
    parser.add_argument("--interval_width", help="衡量未来时间内趋势改变的程度", type=float, default=0.8)
    parser.add_argument("--uncertainty_samples", help="用于估计不确定性区间的模拟抽取数", type=int, default=1000)
    parser.add_argument("--periods", help="向前预测的步数", type=int, default=10)
    parser.add_argument("--freq", help="单位", type=str, default="D")
    parser.add_argument("--holidays", help="节日", type=str, default="None")

    parser.add_argument("--label_name", help="标签列的名称")
    parser.add_argument("--feature_count", help="特征数量")
    parser.add_argument("--sample_ratio", help="测试样本占比", type=float, default=0.3)

    # cluster conf
    parser.add_argument("--app_name", help="the name of spark app", default="prophet")

    # 扩展参数
    parser.add_argument("--no_validation", help="不做交叉验证", action="store_true")
    parser.add_argument("--outlier_handling", help="异常值处理", action="store_true")
    parser.add_argument("--group_field_name", help="分组字段名称", type=str, default="null")
    parser.add_argument("--base_multiple", help="训练数据是预测数据的倍数", type=int, default=12)

    args = parser.parse_args()
    print("========> args:", args)
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



def check_data_null(sc, data, args):
    flag = False
    try:
        count = data[args.label_name].count()
        null_ratio_group_field = sum(pd.isna(data[args.group_field_name])) / count
        null_ratio_label = sum(pd.isna(data[args.label_name])) / count

        schema_list = data.columns.to_list()
        schema_list.remove(args.label_name)
        schema_list.remove(args.group_field_name)
        null_ratio_date = 0
        if len(schema_list) > 1:
            null_ratio_date = sum(pd.isna(data[schema_list[1]])) / count
        elif len(schema_list) == 1:
            null_ratio_date = sum(pd.isna(data[schema_list[0]])) / count

        print("========> cellid空值占比 " + str(null_ratio_group_field))
        print("========> x空值占比 " + str(null_ratio_date))
        print("========> y空值占比 " + str(null_ratio_label))
        if null_ratio_group_field > 0.05 or null_ratio_label > 0.05 or null_ratio_date > 0.05:
            save_data.write_data_to_cluster(sc,
                                            args.evaluate + os.sep + constants.ERROR_LOG,
                                            "数据异常-预测数据空值过多", is_text_file=True)
            flag = True
            exit(0)
    except:
        if not flag:
            print("======> 数据格式异常 ")
            save_data.write_data_to_cluster(sc,
                                            args.evaluate + os.sep + constants.ERROR_LOG,
                                            "数据格式异常", is_text_file=True)
        exit(0)


def check_data_num(sc, data, args):
    """ 对数据进行异常校验 """
    # 计算预测秒数
    predict_seconds = 0;
    if "d" == args.freq.lower():
        predict_seconds = args.periods * 24 * 60 * 60
    elif "y" == args.freq.lower():
        predict_seconds = args.periods * 24 * 60 * 60 * 365
    elif "m" == args.freq.lower():
        predict_seconds = args.periods * 24 * 60 * 60 * 30
    elif "min" == args.freq.lower():
        predict_seconds = args.periods * 60
    elif "h" == args.freq.lower():
        predict_seconds = args.periods * 60 * 60
    else:
        predict_seconds = args.periods * 24 * 60 * 60

    if (data["ds"].max().timestamp() - data["ds"].min().timestamp()) < args.base_multiple * predict_seconds:
        print("======> 数据异常，预测数据过少 ")
        save_data.write_data_to_cluster(sc,
                                        args.evaluate + os.sep + constants.ERROR_LOG,
                                        "数据异常-预测数据过少", is_text_file=True)
        exit(0)


def get_holidays(args):
    holidays = eval(args.holidays)
    if holidays is not None:
        assert type(holidays) == list, "非法的holidays！ "
        holidays = pd.concat((pd.DataFrame(holiday) for holiday in holidays))
    return holidays


def get_prophet_model(args):
    """ 模型初始化 """
    return Prophet(
        growth=args.growth,
        changepoint_range=args.changepoint_range,
        yearly_seasonality="auto" if args.yearly_seasonality == "auto" else eval(args.yearly_seasonality),
        weekly_seasonality="auto" if args.weekly_seasonality == "auto" else eval(args.weekly_seasonality),
        daily_seasonality="auto" if args.daily_seasonality == "auto" else eval(args.daily_seasonality),
        holidays=get_holidays(args),
        seasonality_mode=args.seasonality_mode,
        seasonality_prior_scale=args.seasonality_prior_scale,
        holidays_prior_scale=args.holidays_prior_scale,
        changepoint_prior_scale=args.changepoint_prior_scale,
        mcmc_samples=args.mcmc_samples,
        interval_width=args.interval_width,
        uncertainty_samples=args.uncertainty_samples)


def get_standard_data(sc, data_df, args):
    """ 对数据进行处理 """
    schema_list = data_df.columns.to_list()
    # assert len(schema_list) == 2, "目前只支持两列输入"
    # 将列名修改为 ds,y
    try:
        schema_list.remove(args.label_name)
        if args.no_validation:
            schema_list.remove(args.group_field_name)
            if len(schema_list) > 1:
                data_df = data_df.drop(schema_list[0], axis=1)
                schema_list.remove(schema_list[0])
        print("=======> 删除序号以及分组列之后的表头信息 " + str(schema_list))
        data_df = data_df.rename(columns={args.label_name: "y", schema_list[0]: "ds"})
        data_df["ds"] = pd.to_datetime(data_df["ds"])
    except:
        print("======> 数据格式异常 ")
        save_data.write_data_to_cluster(sc,
                                        args.evaluate + os.sep + constants.ERROR_LOG,
                                        "数据格式异常", is_text_file=True)
        exit(0)

    # 对数处理
    data_df_log = copy.deepcopy(data_df)
    data_df_log["y"] = data_df_log["y"].apply(math.log)

    if not args.no_validation:
        # 交叉验证数据
        # data_df["ds"] = pd.to_datetime(data_df["ds"])
        count = data_df.count().array[0]
        data_df = data_df.sort_values(by="ds")
        split_flag = math.floor(count * (1 - args.sample_ratio))
        train_data_df = data_df.iloc[:split_flag]
        test_data_df = data_df.iloc[split_flag:]
        return data_df, data_df_log, train_data_df, test_data_df
    else:
        return data_df, data_df_log, None, None


def standard_fig(fig):
    """ 将图片标准化 """
    fig.set_figwidth(5)
    fig.set_figheight(5)

def standard_fig_component(fig):
    """ 标准化组件分析图 """
    fig.set_figwidth(6)
    fig.set_figheight(5)
    fig.set_tight_layout(True)
    plt.xticks(size=10, rotation=30)
    plt.tight_layout(1.4)


# 定义趋势变化
res_trend = {}

def train_model_by_all_data(args, data_df, sc, suffix=""):
    """ 使用全量数据训练模型并保存结果到hdfs """
    m = get_prophet_model(args)
    m.fit(data_df)
    future = m.make_future_dataframe(periods=args.periods, freq=args.freq)
    future_tmp = m.make_future_dataframe(periods=args.periods, freq=args.freq, include_history=False)
    forecast = m.predict(future)
    forecast_tmp = m.predict(future_tmp)
    # res_fig = m.plot(forecast)
    # component_fig = m.plot_components(forecast)
    splitFlag = forecast["ds"].count() - data_df["ds"].count()
    res_fig = drawing_pic.plot(m, forecast, splitFlag)
    component_fig = drawing_pic.plot_components(m, forecast)

    dir_name = args.tmp_dir
    # 预测结果
    res_path = dir_name + os.sep + "000000"
    res_df = pd.DataFrame(forecast, columns=["ds", "yhat"])
    res_df_tmp = pd.DataFrame(forecast_tmp, columns=["ds", "yhat"])

    if suffix != "" :
        res_df_tmp[args.group_field_name] = suffix[:-1]
        res_df[args.group_field_name] = suffix[:-1]
        tail_sum = res_df[-args.periods:]["yhat"].sum()
        pre_sum = res_df[-args.periods * 2:-args.periods]["yhat"].sum()
        global res_trend
        res_trend[suffix[:-1]] = {"difference": abs(tail_sum - pre_sum), "amplitude": abs(tail_sum - pre_sum) / pre_sum}

    # res_df.to_csv(res_path, index=False, mode="a", header=False)
    res_df_tmp.to_csv(res_path, index=False, mode="a", header=False)
    # 预测结果图
    res_pic_path = dir_name + os.sep + suffix + constants.PREDICT_PIC
    # 成分分析图
    component_pic_path = dir_name + os.sep + suffix + constants.COMPONENT_PIC
    standard_fig(res_fig)
    standard_fig_component(component_fig)
    res_fig.savefig(res_pic_path)
    component_fig.savefig(component_pic_path)
    with open(res_pic_path, "rb") as f1, open(component_pic_path, "rb") as f2:
        data1 = f1.read()
        data2 = f2.read()
        save_data.write_data_to_cluster(sc,
                                        args.evaluate + os.sep + suffix + constants.PREDICT_PIC,
                                        data1, is_text_file=False)
        save_data.write_data_to_cluster(sc,
                                        args.evaluate + os.sep + suffix + constants.COMPONENT_PIC,
                                        data2, is_text_file=False)

    os.remove(res_pic_path)
    os.remove(component_pic_path)


def train_model_by_log_data(args, data_df_log, sc, suffix=""):
    """ 使用log处理的y进行重新训练 """
    m = get_prophet_model(args)
    m.fit(data_df_log)
    future = m.make_future_dataframe(periods=args.periods, freq=args.freq)
    forecast = m.predict(future)
    png_path = args.tmp_dir + os.sep + suffix + constants.PREDICT_LOG_PIC
    # res_fig = m.plot(forecast)
    splitFlag = forecast["ds"].count() - data_df_log["ds"].count()
    res_fig = drawing_pic.plot(m, forecast, splitFlag)

    standard_fig(res_fig)
    res_fig.savefig(png_path)
    with open(png_path, "rb") as f:
        data = f.read()
        save_data.write_data_to_cluster(sc,
                                        args.evaluate + os.sep + suffix + constants.PREDICT_LOG_PIC,
                                        data, is_text_file=False)
    os.remove(png_path)


def cross_validation(args, train_data_df, test_data_df, sc):
    """ 交叉验证求 mape """
    m = get_prophet_model(args)
    m.fit(train_data_df)
    res = pd.merge(pd.DataFrame(m.predict(test_data_df), columns=["ds", "yhat"]),
                   test_data_df, how="inner", on="ds")
    mape = ((res["y"] - res["yhat"]) / res["y"]).apply(abs).sum() / test_data_df.count().array[0]
    save_data.write_data_to_cluster(sc,
                                    args.evaluate + os.sep + constants.EVALUATION_MODEL_NAME,
                                    str({"mape": str(mape)}), is_text_file=True)


def filter_outliers(data):
    """ 异常值过滤 """
    from pyculiarity import detect_ts
    results = detect_ts(data,
                    max_anoms=0.10,
                    alpha=1000,
                    direction='both',
                    only_last=None)["anoms"]
    outliers_arr = results["timestamp"].array
    data = data[data["timestamp"].apply(lambda ele: ele not in outliers_arr)]
    data = data.rename(columns={"value": "y", "timestamp": "ds"})
    return data


def train_model(args, data_df, data_df_log, sc, suffix=""):
    """ 使用原始数据以及log处理之后的数据进行训练 """
    if args.outlier_handling:
        data_df = filter_outliers(data_df)
        data_df_log = filter_outliers(data_df_log)

    train_model_by_all_data(args, data_df, sc, suffix.strip())
    train_model_by_log_data(args, data_df_log, sc, suffix.strip())


def train_model_parallel(args, data_df, data_df_log, sc):
    """ 对需要分组的数据进行并行处理
        note: 并行有很多问题，先暂时串行
    """
    distinct_group_name = {i for i in data_df[args.group_field_name].array}
    # parallel_num = len(distinct_group_name) if len(distinct_group_name) < 10 else 10
    # p = Pool(parallel_num)
    # for group_name in distinct_group_name:
    #     group_data_df = data_df[data_df[args.group_field_name] == group_name].drop(args.group_field_name, axis=1)
    #     group_data_df_log = data_df_log[data_df_log[args.group_field_name] == group_name].drop(args.group_field_name, axis=1)
    #     p.apply_async(train_model, args=(args, group_data_df, group_data_df_log, sc, "-" + group_name))
    #
    # p.close()
    # p.join()
    l = []
    for group_name in distinct_group_name:
        group_data_df = data_df[data_df[args.group_field_name] == group_name].drop(args.group_field_name, axis=1)
        group_data_df_log = data_df_log[data_df_log[args.group_field_name] == group_name].drop(args.group_field_name, axis=1)
        # 对数据做校验
        check_data_num(sc, group_data_df, args)
        l.append((group_name, group_data_df, group_data_df_log))

    for group_name, group_data_df, group_data_df_log in l:
        train_model(args, group_data_df, group_data_df_log, sc, str(group_name) + "-")

    save_data.write_data_to_cluster(sc,
                                    args.evaluate + os.sep + constants.EVALUATION_MODEL_NAME,
                                    str(res_trend), is_text_file=True)


def save_res_to_hdfs(sc, args):
    # 将预测结果写入hdfs
    res_path = args.tmp_dir + os.sep + "000000"

    if args.group_field_name != "null":
        format_res(res_path, args.tmp_dir + os.sep + "000001")
        res_path = args.tmp_dir + os.sep + "000001"

    with open(res_path, "r") as f:
        data = f.read()
        save_data.write_data_to_cluster(sc,
                                    args.output + os.sep + "000000",
                                    data, is_text_file=True)
    os.remove(res_path)


def format_res(path, path1):
    """ 将path的预测结果规范化至 path1 """
    id = 1
    with open(path, "r") as f, open(path1, "w") as f1:
        f1.writelines("序号,时间,预测目标,流量\n")
        for i in f:
            if i != "":
                fields = i.split(",")
                if len(fields) >= 3:
                    f1.writelines("{0},{1},{2},{3}\n".
                                  format(str(id), fields[0].strip(), fields[2].strip(), fields[1].strip()))
                    id += 1
    os.remove(path)


def main():
    args = parse_args()
    spark, executors, sc = init_spark_session(args)
    data_df = get_train_data(args, spark, executors)
    data_df = sparkDF2pandasDF(sc, data_df)
    if not args.group_field_name == "null":
        check_data_null(sc, data_df, args)

    # 数据预处理
    data_df, data_df_log, train_data_df, test_data_df = get_standard_data(sc, data_df, args)

    # 将预测结果图写入临时目录
    args.tmp_dir = "/tmp/" + str(time.time())
    os.mkdir(args.tmp_dir)

    # 如果没有指定分组字段名称，则默认不分组
    if args.group_field_name == "null":
        train_model(args, data_df, data_df_log, sc)
    else:
        train_model_parallel(args, data_df, data_df_log, sc)
    save_res_to_hdfs(sc, args)
    if not args.no_validation:
        # 交叉验证
        cross_validation(args, train_data_df, test_data_df, sc)

    # 删除本地临时目录
    os.rmdir(args.tmp_dir)
    print("=====> over <======")


if __name__ == "__main__":
    main()





