#!-*-coding=utf8-*-

import numpy as np


def extract_field_species(df, column_name):
    """ 提取某列中的包含的类别数 """
    label_type = [i[0] for i in
                  df.select(column_name).distinct().collect()]
    label_type.sort()
    type_count = len(label_type)
    return label_type, type_count


def agg_features_and_one_hot_label(df, features_name, label_name, feature_type = float):
    """ 将多列特征聚合为一列,并且对label进行onehot编码
        :param df
        :param features_name 特征字段列表
        :param label_name label字段名
    """

    label_type, type_count = extract_field_species(df, label_name)

    def _func(row):
        label_arr = np.zeros(type_count)
        rowDict = row.asDict()
        # one-hot 编码
        label_index = label_type.index(rowDict[label_name])
        label_arr[label_index] = 1
        # 将特征构建成数组
        features_arr = [feature_type(rowDict[feature]) for feature in features_name]

        return (features_arr, label_arr.tolist())

    return df.rdd.map(_func), label_type, type_count


