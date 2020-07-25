#!/bin/bash
# export PYTHONPATH=./python3.6.8:$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.4-src.zip:$PYTHONPATH

# export PYSPARK_PYTHON=./python3.6.8/bin/python

hadoop fs -rm -r /home/mnist-test/result3

# pyspark2
spark2-submit \
--master yarn \
--deploy-mode cluster \
--num-executors 10 \
--executor-cores 1 \
--executor-memory 2g \
--archives  ../tf2.2.zip#PyEnv \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=PyEnv/tf2.2/bin/python \
--conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=PyEnv/tf2.2/bin/python \
--conf spark.executorEnv.PYSPARK_PYTHON=PyEnv/tf2.2/bin/python \
--conf spark.executorEnv.PYSPARK_DRIVER_PYTHON=PyEnv/tf2.2/bin/python \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--jars tensorflow-hadoop-1.0-SNAPSHOT.jar \
tfrecord.py \
--batch_size 128 \
--epochs 10 \
--images_labels hdfs://master01.hadoop.dtmobile.cn:8020/home/mnist-test/data/mnist.tfrecord \
--model_dir hdfs://master01.hadoop.dtmobile.cn:8020/home/mnist-test/result3/model \
--export_dir hdfs://master01.hadoop.dtmobile.cn:8020/home/mnist-test/result3/export





