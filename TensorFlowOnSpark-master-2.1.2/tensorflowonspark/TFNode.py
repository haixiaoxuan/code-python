# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.
"""
  此模块为 tensorflow 应用的工具模块
  在每个executor中执行

  主要方法：
      hdfs_path 返回HDFS路径
      start_cluster_server 返回 cluster对象和server对象
      export_saved_model 保存模型，只适用于1.x
      export_saved_model 导出模型

      DataFeed class


This module provides helper functions for the TensorFlow application.
Primarily, these functions help with:
* starting the TensorFlow ``tf.train.Server`` for the node (allocating GPUs as desired, and determining the node's role in the cluster).
* managing input/output data for *InputMode.SPARK*.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

import getpass
import logging

from packaging import version
from six.moves.queue import Empty
from . import compat, marker

logger = logging.getLogger(__name__)


def hdfs_path(ctx, path):
  """
  将 path 转换为 HDFS 绝对路径
  Convenience function to create a Tensorflow-compatible absolute HDFS path from relative paths

  Args:
    :ctx: TFNodeContext containing the metadata specific to this node in the cluster.
    :path: path to convert

  Returns:
    An absolute path prefixed with the correct filesystem scheme.
  """
  #  All Hadoop-Compatible File System Schemes (as of Hadoop 3.0.x):
  HADOOP_SCHEMES = ['adl://',
                    'file://',
                    'hdfs://',
                    'oss://',
                    's3://',
                    's3a://',
                    's3n://',
                    'swift://',
                    'viewfs://',
                    'wasb://']
  if (any(path.startswith(scheme) for scheme in HADOOP_SCHEMES)):
    # absolute path w/ scheme, just return as-is
    return path
  elif path.startswith("/"):
    # absolute path w/o scheme, just prepend w/ defaultFS
    return ctx.defaultFS + path
  else:
    # relative path, prepend defaultFS + standard working dir
    if ctx.defaultFS.startswith("hdfs://") or ctx.defaultFS.startswith("viewfs://"):
      return "{0}/user/{1}/{2}".format(ctx.defaultFS, getpass.getuser(), path)
    elif ctx.defaultFS.startswith("file://"):
      return "{0}/{1}/{2}".format(ctx.defaultFS, ctx.working_dir[1:], path)
    else:
      logger.warn("Unknown scheme {0} with relative path: {1}".format(ctx.defaultFS, path))
      return "{0}/{1}".format(ctx.defaultFS, path)


def start_cluster_server(ctx, num_gpus=1, rdma=False):
  """
  在executor中执行，启动 tensorflow 应用程序，并对GPU设备进行设置，否则使用cpu
  对 tensorflow2.X 是过时的，返回 (cluster_spec, server)

  Function that wraps the creation of TensorFlow ``tf.train.Server`` for a node in a distributed TensorFlow cluster.
  This is intended to be invoked from within the TF ``map_fun``, replacing explicit code to instantiate ``tf.train.ClusterSpec``
  and ``tf.train.Server`` objects.
  DEPRECATED for TensorFlow 2.x+

  Args:
    :ctx: TFNodeContext containing the metadata specific to this node in the cluster.
    :num_gpu: number of GPUs desired
    :rdma: boolean indicating if RDMA 'iverbs' should be used for cluster communications.

  Returns:
    A tuple of (cluster_spec, server)
  """
  import os
  import tensorflow as tf
  import time
  from . import gpu_info

  if version.parse(tf.__version__) >= version.parse("2.0.0"):
    raise Exception("DEPRECATED: Use higher-level APIs like `tf.keras` or `tf.estimator`")

  logging.info("{0}: ======== {1}:{2} ========".format(ctx.worker_num, ctx.job_name, ctx.task_index))
  cluster_spec = ctx.cluster_spec
  logging.info("{0}: Cluster spec: {1}".format(ctx.worker_num, cluster_spec))

  if compat.is_gpu_available() and num_gpus > 0:
    # compute my index relative to other nodes placed on the same host (for GPU allocation)
    # 在相同的机器上寻找GPU
    my_addr = cluster_spec[ctx.job_name][ctx.task_index]
    my_host = my_addr.split(':')[0]
    flattened = [v for sublist in cluster_spec.values() for v in sublist]
    local_peers = [p for p in flattened if p.startswith(my_host)]
    my_index = local_peers.index(my_addr)

    # GPU
    gpu_initialized = False
    retries = 3
    while not gpu_initialized and retries > 0:
      try:
        # override PS jobs to only reserve one GPU
        if ctx.job_name == 'ps':
          num_gpus = 0

        # Find a free gpu(s) to use
        gpus_to_use = gpu_info.get_gpus(num_gpus, my_index)
        gpu_prompt = "GPU" if num_gpus == 1 else "GPUs"
        logging.info("{0}: Using {1}: {2}".format(ctx.worker_num, gpu_prompt, gpus_to_use))

        # Set GPU device to use for TensorFlow
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use

        # Create a cluster from the parameter server and worker hosts.
        cluster = tf.train.ClusterSpec(cluster_spec)

        # Create and start a server for the local task.
        if rdma:
          server = tf.train.Server(cluster, ctx.job_name, ctx.task_index, protocol="grpc+verbs")
        else:
          server = tf.train.Server(cluster, ctx.job_name, ctx.task_index)
        gpu_initialized = True
      except Exception as e:
        print(e)
        logging.error("{0}: Failed to allocate GPU, trying again...".format(ctx.worker_num))
        retries -= 1
        time.sleep(10)
    if not gpu_initialized:
      raise Exception("Failed to allocate GPU")
  else:
    # CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    logging.info("{0}: Using CPU".format(ctx.worker_num))

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec(cluster_spec)

    # Create and start a server for the local task.
    server = tf.train.Server(cluster, ctx.job_name, ctx.task_index)

  return (cluster, server)


def next_batch(mgr, batch_size, qname='input'):
  """*DEPRECATED*. Use TFNode.DataFeed class instead."""
  raise Exception("DEPRECATED: Use TFNode.DataFeed class instead")



def export_saved_model(sess, export_dir, tag_set, signatures):
  """
  导出TF模型根据提供的参数
  在 TF2.x 过时

  The caller specifies the saved_model signatures in a simplified python dictionary form, as follows::

    signatures = {
      'signature_def_key': {
        'inputs': { 'input_tensor_alias': input_tensor_name },
        'outputs': { 'output_tensor_alias': output_tensor_name },
        'method_name': 'method'
      }
    }

  And this function will generate the `signature_def_map` and export the saved_model.
  DEPRECATED for TensorFlow 2.x+.

  Args:
    :sess: a tf.Session instance
    :export_dir: path to save exported saved_model
    :tag_set: string tag_set to identify the exported graph
    :signatures: simplified dictionary representation of a TensorFlow signature_def_map

  Returns:
    A saved_model exported to disk at ``export_dir``.
  """
  import tensorflow as tf

  if version.parse(tf.__version__) >= version.parse("2.0.0"):
    raise Exception("DEPRECATED: Use TF provided APIs instead.")

  g = sess.graph
  g._unsafe_unfinalize()           # https://github.com/tensorflow/serving/issues/363
  builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

  logging.info("===== signatures: {}".format(signatures))
  signature_def_map = {}
  for key, sig in signatures.items():
    signature_def_map[key] = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={name: tf.saved_model.utils.build_tensor_info(tensor) for name, tensor in sig['inputs'].items()},
        outputs={name: tf.saved_model.utils.build_tensor_info(tensor) for name, tensor in sig['outputs'].items()},
        method_name=sig['method_name'] if 'method_name' in sig else key)
  logging.info("===== signature_def_map: {}".format(signature_def_map))
  builder.add_meta_graph_and_variables(
      sess,
      tag_set.split(','),
      signature_def_map=signature_def_map,
      clear_devices=True)
  g.finalize()
  builder.save()


def batch_results(mgr, results, qname='output'):
  """*DEPRECATED*. Use TFNode.DataFeed class instead."""
  raise Exception("DEPRECATED: Use TFNode.DataFeed class instead")


def terminate(mgr, qname='input'):
  """*DEPRECATED*. Use TFNode.DataFeed class instead."""
  raise Exception("DEPRECATED: Use TFNode.DataFeed class instead")


class DataFeed(object):
  """This class manages the *InputMode.SPARK* data feeding process from the perspective of the TensorFlow application.

  Args:
    :mgr: TFManager instance associated with this Python worker process. python分布式多进程管理.
    :train_mode: boolean indicating if the data feed is expecting an output response (e.g. inferencing).
    :qname_in: *INTERNAL_USE*   输入队列名称
    :qname_out: *INTERNAL_USE*  输出队列名称
    :input_mapping: *For Spark ML Pipelines only*.  Dictionary of input DataFrame columns to input TensorFlow tensors.
    input_mapping: 相当于是 tensorflow 的元数据信息， dataframe columns
  """
  def __init__(self, mgr, train_mode=True, qname_in='input', qname_out='output', input_mapping=None):

    self.mgr = mgr
    self.train_mode = train_mode
    self.qname_in = qname_in
    self.qname_out = qname_out
    self.done_feeding = False
    self.input_tensors = [tensor for col, tensor in sorted(input_mapping.items())] if input_mapping is not None else None

  def next_batch(self, batch_size):
    """ 得到一个批次的训练数据从输入队列中
    Gets a batch of items from the input RDD.
    If multiple tensors are provided per row in the input RDD, e.g. tuple of (tensor1, tensor2, ..., tensorN) and:
    * no ``input_mapping`` was provided to the DataFeed constructor, this will return an array of ``batch_size`` tuples,
      and the caller is responsible for separating the tensors.
    * an ``input_mapping`` was provided to the DataFeed constructor, this will return a dictionary of N tensors,
      with tensor names as keys and arrays of length ``batch_size`` as values.

    Note: if the end of the data is reached, this may return with fewer than ``batch_size`` items.
    Args:
      :batch_size: number of items to retrieve.
    Returns:
      A batch of items or a dictionary of tensors.
    """
    logger.debug("next_batch() invoked")
    queue = self.mgr.get_queue(self.qname_in)   # 得到 input 队列
    # 如果有输入input_tensor，则使用输入的tensor字典，否则为空列表
    tensors = [] if self.input_tensors is None else {tensor: [] for tensor in self.input_tensors}
    count = 0
    while count < batch_size:
      item = queue.get(block=True)
      if item is None:
        # 数据为空结束 Feed
        logger.info("next_batch() got None")
        queue.task_done()
        self.done_feeding = True
        break
      elif type(item) is marker.EndPartition:
        # 收到结束mark， End of Partition
        logger.info("next_batch() got EndPartition")
        queue.task_done()
        if not self.train_mode and count > 0:   # 如果train_mode 是false 直接跳出循环
          break
      else:
        # 正常迭代
        if self.input_tensors is None:
          tensors.append(item)
        else:
          for i in range(len(self.input_tensors)):
            tensors[self.input_tensors[i]].append(item[i])
        count += 1
        queue.task_done()
    logger.debug("next_batch() returning {0} items".format(count))
    return tensors

  def should_stop(self):
    """Check if the feed process was told to stop (by a call to ``terminate``)."""
    return self.done_feeding

  def batch_results(self, results):
    """ 推一个批次的输出结果到输出队列
    Push a batch of output results to the Spark output RDD of ``TFCluster.inference()``.

    Note: this currently expects a one-to-one mapping of input to output data, so the length of the ``results`` array should match the length of
    the previously retrieved batch of input data.
    Args:
      :results: array of output data for the equivalent batch of input data.
    """
    logger.debug("batch_results() invoked")
    queue = self.mgr.get_queue(self.qname_out)
    for item in results:
      queue.put(item, block=True)
    logger.debug("batch_results() returning data")

  def terminate(self):
    """Terminate data feeding early.
    因为tensorflow通常可以在与训练无关的情况下终止
    Since TensorFlow applications can often terminate on conditions unrelated to the training data (e.g. steps, accuracy, etc),
    this method signals the data feeding process to ignore any further incoming data.
    Note that Spark itself does not have a mechanism
    to terminate an RDD operation early, so the extra partitions will still be sent to the executors (but will be ignored).
    Because of this, you should size your input data accordingly to avoid excessive overhead.

    请注意，Spark本身没有提前终止RDD操作的机制，所以额外的分区仍然会发送给执行者
    因此，您应该相应地调整输入数据的大小，以避免过多的开销。
    """
    logger.info("terminate() invoked")  # 停止调用
    self.mgr.set('state', 'terminating')

    # 将队列中元素取空
    # drop remaining items in the queue
    queue = self.mgr.get_queue(self.qname_in)
    count = 0
    done = False
    while not done:
      try:
        queue.get(block=True, timeout=5)
        queue.task_done()
        count += 1
      except Empty:
        logger.info("dropped {0} items from queue".format(count))
        done = True
