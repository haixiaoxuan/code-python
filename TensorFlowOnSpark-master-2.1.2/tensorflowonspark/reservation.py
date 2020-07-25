# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.
"""This module contains client/server methods to manage node reservations during TFCluster startup."""

"""
  管理 TFCluster 启动期间 node 的注册信息
  包含 client/server 的一些方法
  
  msg属于客户端发给服务端的消息，服务端通过判断消息类型来做处理
  msg{
    type: REG|QUERY|QINFO|STOP
    data: reservations[]里面的内容
  }
  
    主要类：
        Reservations    保存节点注册信息
        Server    服务端，等待客户端来注册
        Client
        
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

import logging
import os
import pickle
import select
import socket
import struct
import sys
import threading
import time

from . import util

logger = logging.getLogger(__name__)

TFOS_SERVER_PORT = "TFOS_SERVER_PORT"
TFOS_SERVER_HOST = "TFOS_SERVER_HOST"
BUFSIZE = 1024
MAX_RETRIES = 3 # 最大尝试次数


class Reservations:
  """
  线程安全 存储节点注册信息
  Thread-safe store for node reservations.
  Args:
    :required: 期望的节点数量
  """

  def __init__(self, required):
    self.required = required
    self.lock = threading.RLock() # 递归锁
    self.reservations = [] # 已经注册的节点信息

  def add(self, meta):
    """增加一个节点注册
    Args:
      :meta: a dictonary of metadata about a node
    """
    with self.lock:
      self.reservations.append(meta)

  def done(self):
    """如果注册节点数大于等于需要节点数，则返回true"""
    with self.lock:
      return len(self.reservations) >= self.required

  def get(self):
    """获取注册节点列表"""
    with self.lock:
      return self.reservations

  def remaining(self):
    """还差多少节点"""
    with self.lock:
      return self.required - len(self.reservations)


class MessageSocket(object):
  """
  基类，提供收发消息的方法.
  解决tcp粘包问题
  """
  def receive(self, sock):
    """Receive a message on ``sock``."""
    msg = None
    data = b''
    recv_done = False
    recv_len = -1
    while not recv_done:
      buf = sock.recv(BUFSIZE)
      if buf is None or len(buf) == 0:
        raise Exception("socket closed")
      if recv_len == -1:
        recv_len = struct.unpack('>I', buf[:4])[0]  # 获取消息长度
        data += buf[4:]
        recv_len -= len(data)
      else:
        data += buf
        recv_len -= len(buf)
      recv_done = (recv_len == 0)

    msg = pickle.loads(data)
    return msg

  def send(self, sock, msg):
    """Send ``msg`` to destination ``sock``."""
    data = pickle.dumps(msg)
    buf = struct.pack('>I', len(data)) + data
    sock.sendall(buf)


class Server(MessageSocket):
  """
  socket server
  Args:
    :count: expected number of nodes in the cluster.
  """
  reservations = None             #: List of reservations managed by this server.
  done = False                    #: boolean indicating if server should be shutdown.

  def __init__(self, count):
    assert count > 0, "Expected number of reservations should be greater than zero"
    self.reservations = Reservations(count)


  def await_reservations(self, sc, status={}, timeout=600):
    """
    一直阻塞，直到所有的node都注册完成
    status: 状态信息
    timeout: 超时时间600s
    """
    """Block until all reservations are received."""
    timespent = 0
    while not self.reservations.done(): # 只要没有注册完成
      logger.info("waiting for {0} reservations".format(self.reservations.remaining())) # 打印剩余没有注册数量
      # check status flags for any errors
      if 'error' in status: # 如果有错误消息，则程序退出
        sc.cancelAllJobs()
        sc.stop()
        sys.exit(1)
      time.sleep(1)
      timespent += 1
      if (timespent > timeout):
        raise Exception("timed out waiting for reservations to complete")
    logger.info("all reservations completed")
    return self.reservations.get()


  def _handle_message(self, sock, msg):
    """
    处理消息
    """
    logger.debug("received: {0}".format(msg))
    msg_type = msg['type'] # 消息类型

    if msg_type == 'REG':
      self.reservations.add(msg['data'])  # 添加
      MessageSocket.send(self, sock, 'OK')
    elif msg_type == 'QUERY':
      MessageSocket.send(self, sock, self.reservations.done()) # 查询
    elif msg_type == 'QINFO':
      rinfo = self.reservations.get()
      MessageSocket.send(self, sock, rinfo) # 返回 reservations[]
    elif msg_type == 'STOP': # 停止
      logger.info("setting server.done")
      MessageSocket.send(self, sock, 'OK')
      self.done = True
    else:
      MessageSocket.send(self, sock, 'ERR')

  def start(self):
    """
    开启一个后台线程进行监听，返回监听的主机端口
    Start listener in a background thread
    Returns:
      address of the Server as a tuple of (host, port)
    """
    server_sock = self.start_listening_socket()

    # hostname may not be resolvable but IP address probably will be
    host = self.get_server_ip()
    port = server_sock.getsockname()[1]
    addr = (host, port)
    logger.info("listening for reservations at {0}".format(addr))

    def _listen(self, sock):
      CONNECTIONS = []  # 连接列表
      CONNECTIONS.append(sock)

      while not self.done: # server没有结束
        read_socks, write_socks, err_socks = select.select(CONNECTIONS, [], [], 60)
        for sock in read_socks:
          if sock == server_sock:
            client_sock, client_addr = sock.accept() # 监听客户端如果有客户端进行连接，则打印信息
            CONNECTIONS.append(client_sock)
            logger.debug("client connected from {0}".format(client_addr))
          else:
            try:
              msg = self.receive(sock)
              self._handle_message(sock, msg)   # 处理客户端发来的消息
            except Exception as e:
              logger.debug(e)
              sock.close()
              CONNECTIONS.remove(sock)

      server_sock.close()

    t = threading.Thread(target=_listen, args=(self, server_sock))
    t.daemon = True
    t.start()
    return addr   # 返回监听的主机端口

  def get_server_ip(self): # 获取ip
    return os.getenv(TFOS_SERVER_HOST) if os.getenv(TFOS_SERVER_HOST) else util.get_ip_address()

  def start_listening_socket(self):
    # 从环境变量中获取端口
    port_number = int(os.getenv(TFOS_SERVER_PORT)) if os.getenv(TFOS_SERVER_PORT) else 0
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(('', port_number))
    server_sock.listen(10)
    return server_sock

  def stop(self):
    """ 结束监听 Stop the Server's socket listener."""
    self.done = True


class Client(MessageSocket):
  """
  Client to register and await node reservations.
  Args:
    :server_addr: a tuple of (host, port) pointing to the Server. 连接的服务器地址
  """
  sock = None                   #: socket to server TCP connection
  server_addr = None            #: address of server

  def __init__(self, server_addr):
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.sock.connect(server_addr)
    self.server_addr = server_addr
    logger.info("connected to server at {0}".format(server_addr)) # 连接服务端

  def _request(self, msg_type, msg_data=None):
    """工具函数：打包 msg， 主要是 type data """
    """Helper function to wrap msg w/ msg_type."""
    msg = {}
    msg['type'] = msg_type
    if msg_data:
      msg['data'] = msg_data

    done = False
    tries = 0
    while not done and tries < MAX_RETRIES:   # 默认尝试三次
      try:
        MessageSocket.send(self, self.sock, msg)
        done = True
      except socket.error as e:
        tries += 1
        if tries >= MAX_RETRIES:
          raise
        print("Socket error: {}".format(e))
        self.sock.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(self.server_addr)

    logger.debug("sent: {0}".format(msg))
    resp = MessageSocket.receive(self, self.sock)   # 接受服务器返回的消息，并返回
    logger.debug("received: {0}".format(resp))
    return resp

  def close(self):
    """Close the client socket."""
    self.sock.close()

  def register(self, reservation):
    """ 向服务器注册节点信息 """
    """Register ``reservation`` with server."""
    resp = self._request('REG', reservation)  # 这是添加节点的信息
    return resp

  def get_reservations(self):
    """ 得到当前所有节点的注册信息 """
    """Get current list of reservations."""
    cluster_info = self._request('QINFO')
    return cluster_info

  def await_reservations(self):
    """ 等待所有的节点注册完成，然后返回集群信息 """
    """Poll until all reservations completed, then return cluster_info."""
    done = False
    while not done:
      done = self._request('QUERY')
      time.sleep(1)
    return self.get_reservations()

  def request_stop(self):
    """Request server stop."""
    resp = self._request('STOP')
    return resp
