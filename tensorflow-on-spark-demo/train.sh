#!/bin/bash

hadoop fs -rm -r /home/mnist-test/result/
#export PYSPARK_PYTHON=/usr/bin/python3
#export HADOOP_HDFS_HOME=/etc/hadoop/conf.cloudera.hdfs/

export LIB_HDFS=/opt/cloudera/parcels/CDH-5.13.3-1.cdh5.13.3.p0.2/lib64
export LIB_JVM=/usr/lib/java/jdk1.8.0_144/jre/lib/amd64/server
spark2-submit \
    --master yarn --deploy-mode client \
    --num-executors 3 --executor-memory 6G \
	--executor-cores 1 \
	--conf spark.yarn.maxAppAttempts=1 \
	--conf spark.dynamicAllocation.enabled=false \
	--conf spark.pyspark.driver.python=/usr/bin/python3 \
    --conf spark.pyspark.python=/usr/bin/python3 \
	--conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_JVM:$LIB_HDFS \
    --py-files softmax_dist_pipeline.py \
    softmax_pipeline.py \
	--model_dir hdfs://172.30.5.211:8020/home/mnist-test/result/model \
	--export_dir hdfs://172.30.5.211:8020/home/mnist-test/result/pb \
	--output /home/mnist-test/result/predict \
    --train_path "{'type':'hdfs','path':'/home/mnist-test/data/train'}" \
	--evaluate /home/mnist-test/result/evaluate \
	--inference_mode signature \
	--train \
	--inference_output predictions \
	--steps 30 \
	--label_name label \
	--feature_count 784 \
	--sample_ratio 0.2 \
	--layers_active '2000,Tanh|2000,Tanh|1000,Tanh' \
    --gradient GradientDescent \
    --loss SoftmaxCrossEntropy \
    --activation Softmax \
	--learning_rate 0.2 \
	--batch_size 100 \
	--epochs 1


# sparksql 并行度问题，导致程序oom

# 如果需要用到tfrecord
#!/bin/bash
spark2-submit \
--master yarn \
--deploy-mode client \
--queue default \
--num-executors 4 \
--executor-memory 3G \
--jars /home/etluser/xiexiaoxuan/xiexiaoxuan-test/tensorflow/tfos/tensorflow-hadoop-1.0-SNAPSHOT.jar \
--py-files mnist_dist_pipeline.py \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
mnist_spark_pipeline.py \
--model_dir hdfs://master01.hadoop.dtmobile.cn:8020/home/mnist-test/tfr/model/ \
--export_dir hdfs://master01.hadoop.dtmobile.cn:8020/home/mnist-test/tfr/export/ \
--tfrecord_dir hdfs://master01.hadoop.dtmobile.cn:8020/home/mnist-test/tfr/tfrecord \
--format csv \
--images /home/mnist-test/tfr/images/ \
--labels /home/mnist-test/tfr/labels \
--inference_mode signature \
--train \
