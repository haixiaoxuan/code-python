from pyspark.ml.clustering import GaussianMixture
from pyspark.sql import SparkSession

"""
    高斯混合模型
"""

spark = SparkSession\
    .builder\
    .appName("GaussianMixtureExample")\
    .getOrCreate()

dataset = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")

gmm = GaussianMixture().setK(2).setSeed(538009335)
model = gmm.fit(dataset)

print("Gaussians shown as a DataFrame: ")
model.gaussiansDF.show(truncate=False)
