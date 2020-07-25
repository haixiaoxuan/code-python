#!-*-coding=utf8-*-

def _cal_indicators(target_counts):
    """ 精准率、召回率 计算 """

    def _cal(row):
        tmp = {}
        key = row[0]  # 预测值
        itera = row[1]  # 真实值
        true_count = 0  # 预测值为 1 的样本中真实值为 1 的个数
        predict_count = len(itera)  # 计算预测值为 1 的个数
        for value in itera:
            if key == value:
                true_count += 1
        if target_counts[key] != 0 and predict_count != 0:
            recall = true_count / target_counts[key]
            precise = true_count / predict_count
        else:
            recall = 0
            precise = 0

        tmp[key] = {"recall": recall, "precise": precise}
        return tmp

    return _cal


def cal_recall_and_precise(rdd):
    """ 每个分类的精准率和召回率计算
        :param rdd  row -> tuple(y, predict)
    """
    target_counts = rdd.countByKey()  # 计算真实值为 1 的个数
    final_result_tmp = rdd.map(lambda row: (row[1], row[0])) \
        .groupByKey().map(_cal_indicators(target_counts)).collect()
    return final_result_tmp
