

"""
    signal.SIGHUP   # 连接挂断;
    signal.SIGILL   # 非法指令;
    signal.SIGINT   # 终止进程（ctrl+c）;
    signal.SIGTSTP  # 暂停进程（ctrl+z）;
    signal.SIGKILL  # 杀死进程（此信号不能被捕获或忽略）;
    signal.SIGQUIT  # 终端退出;
    signal.SIGTERM  # 终止信号,软件终止信号;
    signal.SIGALRM  # 闹钟信号,由signal.alarm()发起;
    signal.SIGCONT  # 继续执行暂停进程;
    https://www.cnblogs.com/gengyi/p/8667405.html
"""


import signal

time = 10
# 在一个进程中，只能设置一个时钟，如果设置第二个则会覆盖第一个的时间，返回地一个的剩余时间
signal.alarm(time)  # 设置发送SIGALRM信号的定时器

signal.pause() # Wait until a signal arrives。让进程进程暂停，以等待信号（什么信号均可）；也即阻塞进程进行，接收到信号后使进程停止


# sig：拟需处理的信号，处理信号只针对这一种信号起作用sig
# hander：信号处理方案
sig, handler = 1, 2
signal.signal(sig, handler)


