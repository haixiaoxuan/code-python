
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

"""
    多层感知，利用神经元网络
    
"""
spark = SparkSession \
    .builder.appName("multilayer_perceptron_classification_example").getOrCreate()
data = spark.read.format("libsvm") \
    .load("data/mllib/sample_multiclass_classification_data.txt")


layers = [4, 5, 4, 3]
trainer = MultilayerPerceptronClassifier(maxIter=100,
                                         layers=layers,
                                         blockSize=128,
                                         seed=1234)
model = trainer.fit(data)

result = model.transform(data)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")