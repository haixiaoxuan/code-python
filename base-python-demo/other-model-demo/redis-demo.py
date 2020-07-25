import redis

r = redis.Redis("localhost", 6379, password="root")
r.set("name", "xiaoxuan")

