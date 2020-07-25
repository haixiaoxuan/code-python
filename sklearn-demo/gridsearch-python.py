# 本机测试
# import os
# os.environ['PYSPARK_PYTHON'] = 'D:\\myprogram\\anaconda\envs\\tensorflow\\python'
# os.environ['PYSPARK_DRIVER_PYTHON'] = 'D:\\myprogram\\anaconda\envs\\tensorflow\\python'


def test():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import GradientBoostingClassifier
    # from sklearn.model_selection import GridSearchCV
    from spark_sklearn import GridSearchCV
    from pyspark import SparkConf, SparkContext, HiveContext
    from spark_sklearn import Converter
    import time

    start = time.time()
    conf = SparkConf().setAppName("spark-sklearn")
    sc = SparkContext(conf=conf)
    spark = HiveContext(sc)
    path = "/home/data/data_cell_lable_0521_rsrp_five3_all.csv"
    df = spark.read.csv(path, header=True, inferSchema=True)

    converter = Converter(sc)
    df_data = converter.toPandas(df)
    # 也可以直接使用 pandas的DataFrame进行操作

    # inputpath1 = '/home/etluser/xiexiaoxuan/data/data_cell_lable_0521_rsrp_five3_all.csv'
    # df_data = pd.read_csv(inputpath1)
    df_data = df_data.dropna(axis=0, how='any')

    x1 = df_data.drop(['label'], axis=1)
    y1 = df_data['label']

    gbm0 = GradientBoostingClassifier(n_estimators=262,
                                      max_depth=57,
                                      min_samples_split=50,
                                      random_state=10,
                                      subsample=0.7,
                                      learning_rate=0.01)

    pipeline = Pipeline([("standard", StandardScaler()),
                         ("gbdt", gbm0)])

    params = {
        "gbdt__n_estimators": [i for i in range(10, 20)],
        "gbdt__max_depth": [i for i in range(3, 20)]
    }
    grid_search = GridSearchCV(sc,
                               pipeline,
                               param_grid=params,
                               error_score=0,
                               scoring="accuracy",
                               cv=5,
                               n_jobs=10,
                               pre_dispatch="2*n_jobs",
                               return_train_score=False)

    grid_search.fit(x1, y1)
    end = time.time()
    print("总耗时 ：%.2f s" % (end - start))

    print(grid_search.best_estimator_)
    index = grid_search.best_index_
    res = grid_search.cv_results_
    best_score = res["mean_test_score"][index]
    print("===============: " + str(best_score))


if __name__ == "__main__":
    test()
