


# 多分类模型 评估指标
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator()
evaluator.setMetricName("f1|weightedPrecision|weightedRecall|accuracy")
evaluator.evaluate(...)


# 回归模型 评估指标
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator()
evaluator.setMetricName("rmse|mse|mae|r2")
evaluator.evaluate(...)


# 聚类模型 评估指标 (计算轮廓系数)
from pyspark.ml.evaluation import ClusteringEvaluator
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(...)


