import ray


def regular_function():
    return 1


@ray.remote
def remote_function():
    return 1


for _ in range(3):
    regular_function()


# ��������
for _ in range(3):
    remote_function.remote()


# ---------------- ���̼�ͨ��

id = remote_function.remote()
ray.get(id)         # ����id��ȡ����ķ��ؽ��

