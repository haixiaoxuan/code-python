import ray


def regular_function():
    return 1


@ray.remote
def remote_function():
    return 1


for _ in range(3):
    regular_function()


# 并行任务
for _ in range(3):
    remote_function.remote()


# ---------------- 进程间通信

id = remote_function.remote()
ray.get(id)         # 根据id获取任务的返回结果

