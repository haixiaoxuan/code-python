# Copyright 2019 Yahoo Inc / Verizon Media
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

"""
  完成 tensorflow并行化
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

import logging
from . import TFSparkNode
from . import gpu_info, util

logger = logging.getLogger(__name__)


def run(sc, map_fn, tf_args, num_executors):
  """Runs the user map_fn as parallel, independent instances of TF on the Spark executors.

  Args:
    :sc: SparkContext
    :map_fun: user-supplied TensorFlow "main" function
    :tf_args: ``argparse`` args, or command-line ``ARGV``.  These will be passed to the ``map_fun``.
    :num_executors:  ``--num_executors``指定的数量.

  Returns:
    None
  """

  # 得到spark默认的文件系统
  defaultFS = sc._jsc.hadoopConfiguration().get("fs.defaultFS")
  # strip trailing "root" slash from "file:///" to be consistent w/ "hdfs://..."
  if defaultFS.startswith("file://") and len(defaultFS) > 7 and defaultFS.endswith("/"):
    defaultFS = defaultFS[:-1]

  def _run(it):
    # it为executor的编号 [0 - num_executor-1]
    from pyspark import BarrierTaskContext

    for i in it:
      worker_num = i  # executor编号

    # barrier是2.4 新特性，新引入的调度模型
    # 为了将分布式深度学习嵌入进来
    # barrier stage 会为每个task分配一个id，便于task之间交互
    # use BarrierTaskContext to get placement of all nodes
    ctx = BarrierTaskContext.get()
    tasks = ctx.getTaskInfos()
    nodes = [t.address for t in tasks]

    # use the placement info to help allocate GPUs
    num_gpus = tf_args.num_gpus if 'num_gpus' in tf_args else 1   # 拿出用户指定的每个Task的GPU数量，没有指定默认是1
    # 设置环境变量
    # 并且如果有空闲GPU就为本task设置空闲GPU
    util.single_node_env(num_gpus=num_gpus, worker_index=worker_num, nodes=nodes)

    # run the user map_fn
    ctx = TFSparkNode.TFNodeContext()
    ctx.defaultFS = defaultFS
    ctx.worker_num = worker_num
    ctx.executor_id = worker_num    # executor编号
    ctx.num_workers = len(nodes)    # executor数量

    map_fn(tf_args, ctx)

    # return a dummy iterator (since we have to use mapPartitions)
    return [0]

  # [0... num_executors] 每个executor中一个编号
  nodeRDD = sc.parallelize(list(range(num_executors)), num_executors)
  nodeRDD.barrier().mapPartitions(_run).collect()
