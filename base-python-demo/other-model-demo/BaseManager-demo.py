
"""
    multiprocessing.managers 支持将多进程分布到多台机器上去执行
    一个服务进程可以作为调度者，将任务分布到其他多个进程中
    ( 提供一个多进程共享 queue )

"""

# 主服务器
from multiprocessing.managers import BaseManager
import queue
q = queue.Queue()
class QueueManager(BaseManager): pass

QueueManager.register('get_queue', callable=lambda:q)
# 如果address=('',0)  则会在本机随机挑选一个端口
m = QueueManager(address=('127.0.0.1', 50000), authkey='abc'.encode("utf8"))
s = m.get_server()
s.serve_forever()


# 发送数据
from multiprocessing.managers import BaseManager
class QueueManager(BaseManager): pass
QueueManager.register('get_queue')
m = QueueManager(address=('127.0.0.1', 50000), authkey='abc'.encode("utf8"))
m.connect()
queue = m.get_queue()
queue.put('hello')

# 接收数据
from multiprocessing.managers import BaseManager
class QueueManager(BaseManager): pass
QueueManager.register('get_queue')
m = QueueManager(address=('127.0.0.1', 50000), authkey='abc'.encode("utf8"))
m.connect()
queue = m.get_queue()
print(queue.get())
print(m.address)
