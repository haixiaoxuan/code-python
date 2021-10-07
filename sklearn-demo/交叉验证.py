from sklearn.model_selection import KFold, cross_val_score


KFold

# 验证某个模型再训练集上稳定性，输出k个预测精度
# 输入:
#   clf 分类器
#   x   数据
#   cv  交叉验证，可以为k或者 KFold
cross_val_score