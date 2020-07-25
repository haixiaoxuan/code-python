#!-*-coding=utf8-*-

from pyspark import SparkContext, SparkConf, HiveContext
import argparse
from algorithm_utils import save_data

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--app_name", help="app 名称", type=str, default = "load-data-to-hive")
    parser.add_argument("--path", help="csv路径", type=str)
    parser.add_argument("--destination", help=" json类型的str ", type=str)

    args = parser.parse_args()
    print("args:", args)
    return args


def main():
    # init
    args = parse_args()
    sc = SparkContext(conf=SparkConf().setAppName(args.app_name))
    sc.setLogLevel("WARN")
    hiveContext = HiveContext(sc)

    # read csv data
    data_df = hiveContext.read.csv(args.path, header=True, inferSchema=True)

    dest_dict = eval(args.destination)
    assert type(dest_dict) == dict, "please input str(dict)"

    save_data.save_df(data_df,dest_dict)



if __name__ == "__main__":
    main()