
from pyspark.ml.classification import LinearSVC
from pyspark.sql import SparkSession

"""
    线性 SVM (二分类)
"""

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("linearSVC Example")\
        .getOrCreate()

    training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    lsvc = LinearSVC(maxIter=10, regParam=0.1)

    lsvcModel = lsvc.fit(training)

    lsvcModel.coefficients      # 系数
    lsvcModel.intercept         # 截距

