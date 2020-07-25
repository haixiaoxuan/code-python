from queue import Queue
from threading import Thread
import threading
import time

def worker():
    while True:
        item = q.get()
        print(threading.currentThread().name, item)
        q.task_done()

q = Queue()
for i in range(10):
    t = Thread(target=worker)
    t.daemon = True
    t.start()


for item in range(100):
    if(item % 10 == 0):
        q.join()
    time.sleep(1)
    q.put(item)

q.join()
print("========")