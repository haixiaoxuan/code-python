安装：
    如果 pip install tensorflow 之后 进行import 报错
    可能是因为 cpu 和 tensorflow 版本原因

    可以使用python版本3.5.2和3.6.6
    pip install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-0.12.0rc0-cp35-cp35m-win_amd64.whl

    根据 TensorFlow 1.6.0 的发布说明，该版本会使用 AVX 指令，所以可能会在旧 CPU 上不能运行
    pip install tensorflow==1.5
    https://blog.csdn.net/u010099080/article/details/53418159

distributed tensorflow 包含两类job:
    ps server 和 worker server, 每类job 可以包含多个 task,
    https://blog.csdn.net/tiangcs/article/details/85952007
    cluster = tf.train.ClusterSpec({
                    worker:[ip:port,..],ps:[] })
    server = tf.train.Server(cluster, job_name="worker", task_index=0)


tensorflowonspark数据获取：
    TensorFlow Readers和QueueRunners机制直接读取HDFS数据文件，Spark不访问数据
    SparkRDD 数据发送TensorFlow节点，数据通过feed_dict机制传入TensorFlow计算图

===================================================
问题：
    http://www.imooc.com/article/266859
    1. 报错：Run called even after should_stop requested.
        StopAtStepHook 中 last_step 过小所致
        last_step = (total_batch-1) x training_epochs
    2. 运行程序时报错 AttributeError: 'AutoProxy[get_queue]' object has no attribute 'put'
        将每个executor的core设为1
    3. 运行时报错 No such file or directory: 'executor_id'
        需要在spark-submit后面加上
        --conf spark.dynamicAllocation.enabled=false
    4. 保存模型到 hdfs 出错（卡在模型初始化），保存到本地目录没事
        配置环境变量
        https://github.com/yahoo/TensorFlowOnSpark/wiki/GetStarted_YARN
        #export LIB_HDFS=/opt/cloudera/parcels/CDH/lib64                      # for CDH (per @wangyum)
        export LIB_HDFS=$HADOOP_PREFIX/lib/native/Linux-amd64-64              # path to libhdfs.so, for TF acccess to HDFS
        export LIB_JVM=$JAVA_HOME/jre/lib/amd64/server                        # path to libjvm.so

