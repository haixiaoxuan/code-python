
from pyspark.ml.classification import OneVsRest
from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import RFormula
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

"""
    测试：使用 spark GBDT 二分类实现多分类
    note:  Only LogisticRegression and NaiveBayes are supported now.
    
"""

# 构建 SparkSession
spark = SparkSession \
    .builder \
    .appName(" GBDT TEST ") \
    .enableHiveSupport() \
    .getOrCreate()
sc = spark.sparkContext

# 从 HDFS 上读取数据
path = '/home/mnist-test/data/train'
df = spark.read.csv(path, header=True, inferSchema=True)
df = df.dropna()     # 删除空值

# 将数据转换为 features labels
rf = RFormula(formula="label ~ .", featuresCol="features", labelCol="labels")
rf_model = rf.fit(df)
df = rf_model.transform(df).select(["features", "labels"])

# 数据集切分
train_df, test_df = df.randomSplit([0.8, 0.2])

# 构造 GBDT 模型
gbdt = GBTClassifier(maxIter=10,
                     maxDepth=3,
                     labelCol="labels",
                     featuresCol="features")

# 构造 One Vs Rest Classifier.
ovr = OneVsRest(classifier=gbdt)
ovr_model = ovr.fit(train_df)
predict_res = ovr_model.transform(test_df)

# 评估
evaluator = MulticlassClassificationEvaluator(
    labelCol="labels", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predict_res)



