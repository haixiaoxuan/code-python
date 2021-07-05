import os
os.environ["JAVA_HOME"] = r"C:\xiexiaoxuan\study_app\jdk"
os.environ["SPARK_HOME"] = r"C:\xiexiaoxuan\study_app\spark-2.3.0-bin-hadoop2.7\spark-2.3.0-bin-hadoop2.7"

import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession


memory = '5g'
pyspark_submit_args = ' --driver-memory ' + memory + ' pyspark-shell'
os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args
os.environ["PYSPARK_PYTHON"] = r"C:\Users\xiexiaoxuan\AppData\Local\Programs\Python\Python36\python.exe"


conf = SparkConf().setAppName('test_parquet')
sc = SparkContext('local[*]', 'test', conf=conf)
spark = SparkSession(sc)


path_ad = r"C:\Users\xiexiaoxuan\Desktop\腾讯广告算法\2017腾讯赛数据集\final\ad.csv"
path_app_categories = r"C:\Users\xiexiaoxuan\Desktop\腾讯广告算法\2017腾讯赛数据集\final\app_categories.csv"
path_position = r"C:\Users\xiexiaoxuan\Desktop\腾讯广告算法\2017腾讯赛数据集\final\position.csv"
path_test = r"C:\Users\xiexiaoxuan\Desktop\腾讯广告算法\2017腾讯赛数据集\final\test.csv"
path_train = r"C:\Users\xiexiaoxuan\Desktop\腾讯广告算法\2017腾讯赛数据集\final\train.csv"
path_user = r"C:\Users\xiexiaoxuan\Desktop\腾讯广告算法\2017腾讯赛数据集\final\user.csv"
path_user_app_actions = r"C:\Users\xiexiaoxuan\Desktop\腾讯广告算法\2017腾讯赛数据集\final\user_app_actions.csv"
path_user_installedapps = r"C:\Users\xiexiaoxuan\Desktop\腾讯广告算法\2017腾讯赛数据集\final\user_installedapps.csv"

paths = {
    "ad": path_ad,
    "app_categories": path_app_categories,
    "position": path_position,
    "test": path_test,
    "train": path_train,
    "user": path_user,
    "user_app_actions": path_user_app_actions,
    "user_installedapps": path_user_installedapps,
}


for k, v in paths.items():
    data = spark.read.csv(v, header=True, inferSchema=True)
    print("======== {0} ==========".format(k))
    data.describe().show()







