# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

import logging
import os
import socket
import subprocess
import errno
from socket import error as socket_error
from . import compat, gpu_info

logger = logging.getLogger(__name__)

"""
    工具类：
        single_node_env   为task设置环境变量以及分配GPU
        get_ip_address  获取本机IP
        find_in_path  给环境变量path以及文件名称name，返回文件全路径名
        write_executor_id
        read_executor_id  在工作目录读出executor_id
"""

def single_node_env(num_gpus=1, worker_index=-1, nodes=[]):
  """
  此方法为 task中需要执行的方法，为Hadoop兼容性和GPU分配设置环境变量
  Setup environment variables for Hadoop compatibility and GPU allocation
  work_index: 为此 task 所在的 executor 的 id 信息
  nodes: 为所有 task 的节点信息
  """
  # ensure expanded CLASSPATH w/o glob characters (required for Spark 2.1 + JNI)
  if 'HADOOP_PREFIX' in os.environ and 'TFOS_CLASSPATH_UPDATED' not in os.environ:
      classpath = os.environ['CLASSPATH']
      hadoop_path = os.path.join(os.environ['HADOOP_PREFIX'], 'bin', 'hadoop')
      hadoop_classpath = subprocess.check_output([hadoop_path, 'classpath', '--glob']).decode()
      os.environ['CLASSPATH'] = classpath + os.pathsep + hadoop_classpath
      os.environ['TFOS_CLASSPATH_UPDATED'] = '1'

  if compat.is_gpu_available() and num_gpus > 0:
    # reserve GPU(s), if requested
    if worker_index >= 0 and len(nodes) > 0:
      # compute my index relative to other nodes on the same host, if known
      my_addr = nodes[worker_index]   # 此executor中的task信息
      my_host = my_addr.split(':')[0]
      local_peers = [n for n in nodes if n.startswith(my_host)]   # 拿出本机所有的 task 信息
      my_index = local_peers.index(my_addr)   # 返回此计算机中的所有task的相对index
    else:
      # otherwise, just use global worker index
      my_index = worker_index

    # 获取可用的空闲GPU
    # 分配算法: 根据本机中所有的空闲GPU,每个task分配 num_gpus 个空闲GPU
    gpus_to_use = gpu_info.get_gpus(num_gpus, my_index)
    logger.info("Using gpu(s): {0}".format(gpus_to_use))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use
  else:
    # CPU
    logger.info("Using CPU")
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def get_ip_address():
  """ 获取本机IP"""
  """Simple utility to get host IP address."""
  try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip_address = s.getsockname()[0]
  except socket_error as sockerr:
    if sockerr.errno != errno.ENETUNREACH:
      raise sockerr
    ip_address = socket.gethostbyname(socket.getfqdn())
  finally:
    s.close()

  return ip_address


def find_in_path(path, file):
  """
  从给定的path列表中找出file在那个path下
  Find a file in a given path string."""
  for p in path.split(os.pathsep):
    candidate = os.path.join(p, file)
    if os.path.exists(candidate) and os.path.isfile(candidate):
      return candidate
  return False


def write_executor_id(num):
  """再当前executor工作目录写入executor_id"""
  """Write executor_id into a local file in the executor's current working directory"""
  with open("executor_id", "w") as f:
    f.write(str(num))


def read_executor_id():
  """Read worker id from a local file in the executor's current working directory"""
  with open("executor_id", "r") as f:
    return int(f.read())
