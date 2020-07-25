# -*-coding=utf8-*-

import os

# 路径分隔符
PATH_SEP = os.path.sep


"""  tensorflow 节点名称 """
# 输入层名称
INPUT_LAYER_NAME = "inputs"
# x
INPUT_LAYER_X = "x"
# y
INPUT_LAYER_Y = "y_target"
# argmax(y)
LABEL_NODE_NAME = "label"
# 预测节点名称
PREDICT_NODE_NAME = "prediction"



# 单个pb 文件名称
PB_NAME = "model.pb"
# 复合pb文件夹名称
PB_FOLDER_NAME = "pb"
# 模型评估结果名称
EVALUATION_MODEL_NAME = "evaluate_res.txt"

# 图，参数分离的pb文件输出常量定义
SIG_INPUT = "features"
SIG_OUTPUT = "prediction"


# PMML 模型文件名称
PMML_NAME = "model.pmml"
# pkl 模型文件
PKL_NAME = "model.pkl"
# classification_report
CLASSIFICATION_REPORT = "classification.report"
# confusion matrix
CONFUSION_MATRIX = "confusion.matrix"
# ROC曲线图
ROC_PIC = "roc_auc.png"

# 预测结果图
PREDICT_PIC = "predict.png"
# 自然对数预测结果图
PREDICT_LOG_PIC = "predict_log.png"
# 趋势分解图
COMPONENT_PIC = "component.png"


# 错误日志
ERROR_LOG = "error.log"


