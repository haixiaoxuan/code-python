import pandas as pd
from catboost import CatBoostClassifier


model = CatBoostClassifier(
    task_type="GPU",    # 使用GPU来进行加速
    iterations=2,       # 生成多少颗树（对称树）
    learning_rate=0.03,
    depth=10,
    l2_leaf_reg=3,      # L2正则化系数，防止过拟合
    od_type="Iter",     # 防止过拟合，提早推出训练
    early_stopping_rounds=10,   # 连续10次迭代都不再提升（loss或eval_metric），会终止训练
    n_estimators=1000,  # 解决ml问题的树的最大数量
    one_hot_max_size=2,     # 离散特征类型属性，最大时特征值，超过这个值会采用另外一种编码方式
    loss_function="logloss",    # 损失函数选择 RMSE MAE CrossEntropy
    use_best_model=True,    # 使用最有模型算法
    custom_loss=["AUC", "Accuracy"],
    custom_metric=["Accuracy", "Recall"],
    eval_metric="accuracy"
)

model.fit(train_pool,
          eval_set=test_pool,
          slice=True,   # 不输出任何中间日志
          verbose=True, # 输出日志
          plot=True,    # 显示模型如何学习以及是否开始模型过拟合的可视化界面
          sample_weight=None,   # 输入样本权重
          init_model=None,      # 可以加载训练好的模型参数
          save_snapshot=True,   # 默认每隔600s保存训练快照，以便恢复训练


          )


