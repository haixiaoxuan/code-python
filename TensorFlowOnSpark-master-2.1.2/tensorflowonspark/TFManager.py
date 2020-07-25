# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

from multiprocessing.managers import BaseManager
from multiprocessing import JoinableQueue


class TFManager(BaseManager):
  """
  Python multiprocessing.Manager for distributed, multi-process communication.
  python 分布式多进程管理
  """
  pass

# 全局的Manager，此处三个变量都是全局
# global to each Spark executor's python worker
mgr = None        # TFManager
qdict = {}        # dictionary of queues  --> worker: input | output      ps: control | error
# ps通过control队列来控制进程停止，如果收到一条消息为None，则会设置state为stopped

kdict = {}        # dictionary of key-values  --> 'state': terminating|stopped|running


def _get(key):
  return kdict[key]


def _set(key, value):
  kdict[key] = value


def _get_queue(qname):
  """ 通过队列名称拿到队列 """
  try:
    return qdict[qname]
  except KeyError:
    return None


def start(authkey, queues, mode='local'):
  """Create a new multiprocess.Manager (or return existing one).

  Args:
    :authkey: string authorization key
    :queues: *INTERNAL_USE* 队列名称
    :mode: 'local' indicates that the manager will only be accessible from the same host, otherwise remotely accessible.
    'local'表示只能从同一主机访问管理器，否则将进行远程访问。

  Returns:
    A TFManager instance, which is also cached in local memory of the Python worker process.
  """
  global mgr, qdict, kdict
  qdict.clear()
  kdict.clear()
  for q in queues:
    qdict[q] = JoinableQueue()

  TFManager.register('get_queue', callable=lambda qname: _get_queue(qname))
  TFManager.register('get', callable=lambda key: _get(key))
  TFManager.register('set', callable=lambda key, value: _set(key, value))
  if mode == 'remote':
    # 如果是remote则会再本机随机启动一个端口，可以用来远程访问
    mgr = TFManager(address=('', 0), authkey=authkey)
  else:
    # 如果是local，则只会有一个唯一字符串，不能用来远程访问
    mgr = TFManager(authkey=authkey)
  mgr.start()
  return mgr


def connect(address, authkey):
  """Connect to a multiprocess.Manager.

  Args:
    TFManager的唯一地址，要么是“本地”的唯一连接字符串，要么是远程的(主机、端口)元组
    :address: unique address to the TFManager, either a unique connection string for 'local', or a (host, port) tuple for remote.
    :authkey: string authorization key

  Returns:
    A TFManager instance referencing the remote TFManager at the supplied address.
  """
  TFManager.register('get_queue')
  TFManager.register('get')
  TFManager.register('set')
  m = TFManager(address, authkey=authkey)
  m.connect()
  return m
