#!/bin/bash
# export PYTHONPATH=./python3.6.8:$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.4-src.zip:$PYTHONPATH

export PYSPARK_PYTHON=../tf2.2/bin/python

hadoop fs -rm -r /home/mnist-test/result3

export LIB_HDFS=/opt/cloudera/parcels/CDH-5.13.3-1.cdh5.13.3.p0.2/lib64
export LIB_JVM=/usr/lib/java/jdk1.8.0_144/jre/lib/amd64/server

export HADOOP_HDFS_HOME=/opt/cloudera/parcels/CDH-5.13.3-1.cdh5.13.3.p0.2/lib/hadoop
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HDFS_HOME/lib/native:$JAVA_HOME/jre/lib/amd64/server

export PATH=$LIB_HDFS:$LIB_JVM:$PATH

# pyspark2
spark2-submit \
--master yarn \
--deploy-mode client \
--num-executors 1 \
--executor-cores 1 \
--archives  ../tf2.2.zip#PyEnv \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=PyEnv/tf2.2/bin/python \
--conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=PyEnv/tf2.2/bin/python \
--conf spark.executorEnv.PYSPARK_PYTHON=PyEnv/tf2.2/bin/python \
--conf spark.executorEnv.PYSPARK_DRIVER_PYTHON=PyEnv/tf2.2/bin/python \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--conf spark.executorEnv.LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/cloudera/parcels/CDH-5.13.3-1.cdh5.13.3.p0.2/lib64:$JAVA_HOME/jre/lib/amd64/server" \
--jars tensorflow-hadoop-1.0-SNAPSHOT.jar \
local.py





