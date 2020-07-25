from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

"""
    分类决策树
"""

spark = SparkSession \
    .builder \
    .appName("DecisionTreeClassificationExample") \
    .getOrCreate()

data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
(trainingData, testData) = data.randomSplit([0.7, 0.3])
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

# train
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])
model = pipeline.fit(trainingData)
predictions = model.transform(testData)

predictions.select("prediction", "indexedLabel", "features").show(5)

evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))

treeModel = model.stages[2]

# ==================================================================================

from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

dt = DecisionTreeRegressor(featuresCol="indexedFeatures")
model = dt.fit(trainingData)

# 预测评估
predictions = model.transform(testData)
evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)