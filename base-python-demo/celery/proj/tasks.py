
def redis_test():
    """ 测试 redis """
    import redis
    r = redis.Redis(host='127.0.0.1', port=6379, password="root")
    r.set('foo', 'Bar')
    print(r.get('foo'))


from proj.celery import app


@app.task
def add(a, b):
    return a + b



