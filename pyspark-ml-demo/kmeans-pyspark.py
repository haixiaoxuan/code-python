
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SparkSession


"""
    参数：
        k               分类数
        initMode        初始点的选择，默认是 kmeans|| (kmeans++)
    属性方法：
        clusterCenters()            簇中心
        computeCost(df)             给定数据到最近中心距离的平方和
        model.summary.k             簇的个数
        model.summary.clusterSizes  每个簇的个数
"""
spark = SparkSession \
    .builder \
    .appName("KMeansExample") \
    .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")
df = spark.read.format("libsvm").load("/home/data/cell_building_0520_info_ed.csv")

for i in range(5, 60):
    kmeans = KMeans(maxIter=300).setK(2).setSeed(28)
    model = kmeans.fit(df)
    predictions = model.transform(df)
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    print("===================================")
    print("k = %d - 轮廓系数 ：%f " % (i, silhouette))
    print("====================================")
spark.stop()

# ===================================================================================

""" BisectionKmeans"""
from pyspark.ml.clustering import BisectingKMeans

bkm = BisectingKMeans().setK(2).setSeed(1)
model = bkm.fit(df)

cost = model.computeCost(df)    # 误差平方和
centers = model.clusterCenters()    # 聚类中心

