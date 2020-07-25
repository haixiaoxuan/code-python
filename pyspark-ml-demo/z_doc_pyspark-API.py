from pyspark.sql import SparkSession
from pyspark.sql import Row
import pandas as pd
import numpy as np

# local
import os
os.environ['SPARK_HOME'] = 'D:\\myprogram\\spark-2.3.0-bin-hadoop2.7'

# 创建sparksession
spark = SparkSession \
    .builder \
    .appName("app") \
    .enableHiveSupport() \
    .getOrCreate()

"""
    content
            spark-dataframe
            spark-hive
            spark-postgresql
            spark-pandas
"""



# 创建 spark-dataframe
rdd = spark.sparkContext.parallize()
spark.createDataFrame(rdd, ['col1', 'col2'])

people = rdd.map(lambda p: Row(name=p[0], age=int(p[1])))
schemaPeople = spark.createDataFrame(people)

Record = Row("key", "value")
df = spark.createDataFrame([Record(i, i) for i in range(1, 101)])

from pyspark.sql.types import *
schemaString = "name age"
fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
schema = StructType(fields)
schemaPeople = spark.createDataFrame(people, schema)

# df 操作
df.select("key")
df.select(df["key"], df["value"]+1)
df.filter(df["value"] > 23)
df.groupBy("key").count()

# spark-hive
spark.sql("CREATE TABLE IF NOT EXISTS src (key INT, value STRING) USING hive")
spark.sql("LOAD DATA LOCAL INPATH 'examples/src/main/resources/kv1.txt' INTO TABLE src")
    # DF转换为rdd之后的Row类型对象，可以直接 row.列名 拿到数据

# spark-postgresql
jdbcDF = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:postgresql:dbserver") \
    .option("dbtable", "schema.tablename") \
    .option("user", "username") \
    .option("password", "password") \
    .load()
jdbcDF2 = spark.read \
    .jdbc("jdbc:postgresql:dbserver", "schema.tablename",
          properties={"user": "username", "password": "password"})


# spark-pandas 利用spark中的Array
# 需要手动安装 pyarrow | pip install pyarrow
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
pdf = pd.DataFrame(np.random.rand(100, 3))
df = spark.createDataFrame(pdf)     # 将pandas中的df转换为spark的df
result_pdf = df.select("*").toPandas()

# udf  scalar_pandas_udf 实现高效的udf函数
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import LongType, FloatType
def multiply_func(a, b):
    return a * b
multiply = pandas_udf(multiply_func, returnType=FloatType())    # 也可以通过注解来使用
df = spark.createDataFrame(pd.DataFrame(np.random.random(10, 2), columns=["x", "y"]))
df.select(multiply(col("x"), col("x")).alias("alias")).show()      # 直接使用函数也ok，效率比较低
df.withColumn(...)
# sql udf
spark.udf.register('multiply', multiply)    # 注册为sql语句可用的函数

# grouped_map_pandas_udf    即 聚合函数
from pyspark.sql.functions import pandas_udf, PandasUDFType
df = spark.createDataFrame([(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)],("id", "v"))
@pandas_udf("id long, v double", PandasUDFType.GROUPED_MAP)
def substract_mean(pdf):
    # pdf is a pandas.DataFrame
    v = pdf.v
    return pdf.assign(v=v - v.mean())
df.groupby("id").apply(substract_mean).show()
"""
    	        Scalar	                Grouped map
    udf入参类型	pandas.Series	        pandas.DataFrame
    udf返回类型	pandas.Series	        pandas.DataFrame
    聚合语义	        无	                groupby 的子句
    返回大小	    与输入一致	        rows 和 columns 都可以和入参不同
    返回类型声明	pandas.Series 的 DataType	pandas.DataFrame 的 StructType
"""







