"""
    spark.ml
    主要是针对 spark中的 DataFrame 进行处理的
"""
from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("app") \
    .enableHiveSupport() \
    .getOrCreate()

"""
    content     
        特征处理
        linalg  线性代数
        数据格式
        调优
        
"""

# 特征处理
from pyspark.ml.feature import VectorAssembler      # 将列类型转换为vector
from pyspark.ml.feature import PCA
from pyspark.ml.feature import Normalizer           # 正则化, 对每个样本计算其p-范数
# Normalizer().setP(2).transform(..., {this.p: float("inf")})   使用 float("inf") 替换原来的p

from pyspark.ml.feature import StandardScaler       # 标准化

from pyspark.ml.feature import OneHotEncoder        # one-hot编码 (2.3 是过时的)
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml.feature import StringIndexer        # 将标签类的数据转换为整数索引
                                                    # 默认 出现频率最高的标签为0号 ,如果新数据中出现以前没有出现的标签，则抛异常，
                                                    # 也可以设置忽略 setHandleInvalid("skip")
from pyspark.ml.feature import IndexToString        # 将整数索引转换为标签，直接使用 transform 即可
from pyspark.ml.feature import VectorIndexer        # 将向量类型的列中
                                                    # 解决向量数据集中的类别特征索引。它可以自动识别哪些特征是类别型的，并且将原始值转换为类别索引
from pyspark.ml.feature import RFormula             # 将feature进行聚合成向量 [[feature],label]
                                                    # "label ~ ." | "label ~ col1 +  col2" | "label ~ . - col1"
from pyspark.ml.feature import Binarizer            # 进行 0|1 编码，需指定一个阈值

from pyspark.ml.feature import SQLTransformer       # 定义一条sql语句，使得df根据这条sql进行转换
SQLTransformer(statement="SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__")    # v1-v4 是 df 的字段名

from pyspark.ml.feature import MaxAbsScaler         # 除以最大绝对值，将特征缩放到 [-1,1], 可以保住0元素
from pyspark.ml.feature import MinMaxScaler         # 减去最小值，除以最大值与最小值的差 [0,1]

from pyspark.ml.feature import Imputer              # 缺值补充估计器，默认使用均值来填补

from pyspark.ml.feature import CountVectorizer      # 构建单词向量，用来处理文本数据
from pyspark.ml.feature import PolynomialExpansion  # 多项式扩展
from pyspark.ml.feature import QuantileDiscretizer  # 将特征从大到小排序之后进行分桶


# linalg
from pyspark.ml.linalg import Vectors               # 可以定义向量 Vectors.dense([])



# 数据格式

# libsvm 格式数据加载     label index1:col1 index2:col2
spark.read.format("libsvm").load()      # 会直接将特征聚合 [features],[label]  features类型为vector


# 调优
from pyspark.ml.tuning import ParamGridBuilder              # 交叉验证的参数
from pyspark.ml.tuning import TrainValidationSplit          # 指定split的交叉验证
from pyspark.ml.tuning import CrossValidator                # k折交叉验证，并且实现网格搜索




VectorAssembler, PCA, Normalizer, StandardScaler, OneHotEncoder, StringIndexer
IndexToString, VectorIndexer, RFormula, Binarizer, Vectors, MaxAbsScaler, MinMaxScaler
Imputer, CountVectorizer, ParamGridBuilder,TrainValidationSplit,CrossValidator
OneHotEncoderEstimator,PolynomialExpansion

