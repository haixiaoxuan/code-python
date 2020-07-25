#!-*-coding=utf8-*-

import gevent
import multiprocessing
from multiprocessing import Pool

"""
    使用携程注意点：
        需要加上 
                import socket
                import threading
                from gevent import socket,threading,monkey
                monkey.patch_all()
        把标准库中的thread/socket等给替换掉.这样我们在后面使用socket的时候能够跟寻常一样使用,
        无需改动不论什么代码,可是它变成非堵塞的了
"""

def say():
    print(multiprocessing.current_process().name)


def multi_process():
    """ 因为创建子进程通过 fork ，windows 不支持 """

    process = multiprocessing.Process(target=say)
    process.start()


def gevent_test():
    """ 携程 """

    from gevent import monkey
    monkey.patch_all()

    def func1():
        print('\033[31;1m func1 start ... \033[0m')
        gevent.sleep(2)
        print('\033[31;1m func1 ending ... \033[0m')

    def func2(name):
        print('\033[32;1m func2 start ... \033[0m', name)
        gevent.sleep(1)
        print('\033[32;1m func2 ending ... \033[0m', name)

    gevent.joinall([
        gevent.spawn(func1),
        gevent.spawn(func2, "xiaoxuan"),
    ])


if __name__ == "__main__":
    # multi_process()
    # gevent_test()

    def print():
        print("==== " + multiprocessing.current_process().name)
    p = Pool(3)
    p.apply_async(print)
    p.close()
    p.join()


