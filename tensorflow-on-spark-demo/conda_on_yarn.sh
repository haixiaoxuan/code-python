# export PYSPARK_PYTHON=./python3.6.8/bin/python

pyspark2 \
--master yarn \
--deploy-mode client \
--num-executors 3 \
--executor-cores 1 \
--archives python3.6.8.zip#PyEnv \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=PyEnv/python3.6.8/bin/python \
--conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=PyEnv/python3.6.8/bin/python \
--conf spark.executorEnv.PYSPARK_PYTHON=PyEnv/python3.6.8/bin/python \
--conf spark.executorEnv.PYSPARK_DRIVER_PYTHON=PyEnv/python3.6.8/bin/python


# note:
# 1. PYSPARK_PYTHON client模式下如果显示指定此变量，
#     会导致此环境变量不能被spark.yarn.appMasterEnv.PYSPARK_PYTHON重写
# 2. 以 client模式运行暂未测试通过...

# 测试成功案例：
spark2-submit \
--master yarn \
--deploy-mode cluster \
--num-executors 3 \
--executor-cores 1 \
--archives  python3.6.8.zip#PyEnv \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=PyEnv/python3.6.8/bin/python \
--conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=PyEnv/python3.6.8/bin/python \
--conf spark.executorEnv.PYSPARK_PYTHON=PyEnv/python3.6.8/bin/python \
--conf spark.executorEnv.PYSPARK_DRIVER_PYTHON=PyEnv/python3.6.8/bin/python \
test.py


# 测试
def run(it):
  import tensorflow as tf
  return tf.__version__

rdd = sc.parallelize([1,2,3], 3)
rdd.map(run).collect()
