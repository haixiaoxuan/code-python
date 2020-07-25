#!-*-coding=utf8-*-

from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark import SparkContext, SparkConf, HiveContext, Row
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

"""
    目前仅支持二分类
"""

sc = SparkContext(conf=SparkConf().setAppName("GBDT"))
sc.setLogLevel("WARN")
spark = HiveContext(sc)

# 1.读取数据
feat = ",".join(["col" + str(i) for i in range(784)])
data = spark.sql("select {0},if(label={1},1,0) from mnist".format(feat, 1))
# 2.构造训练数据集
dataSet = data.rdd.map(list)
trainingSet = dataSet.map(list).map(lambda x: Row(label=x[-1], features=Vectors.dense(x[:-1]))).toDF()
train, test = trainingSet.randomSplit([0.7, 0.3])

# 3.使用GBDT进行训练
stringIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel") \
    .fit(trainingSet)
featureIndexer = VectorIndexer(inputCol="features",
                               outputCol="indexedFeatures", maxCategories=4).fit(trainingSet)
gbdt = GBTClassifier(maxIter=10, maxDepth=6, labelCol="indexedLabel", featuresCol="indexedFeatures", seed=42)
pipeline = Pipeline(stages=[stringIndexer, featureIndexer, gbdt])

model = pipeline.fit(train)
predictions = model.transform(test)

# 评估
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# ========================================================================================
"""
    回归树
"""

from pyspark.ml.regression import GBTRegressor

gbt = GBTRegressor(featuresCol="indexedFeatures", maxIter=10)
