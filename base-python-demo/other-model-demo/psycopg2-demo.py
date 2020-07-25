import psycopg2

"""
    可以用来操作postgresql 数据库(多个线程可以共享相同的连接)
"""

# 创建连接对象
conn = psycopg2.connect(database="volcanoweb",
                        user="volcanoweb",
                        password="volcanoweb",
                        host="172.30.4.106",
                        port="5432")
# 创建指针对象
cur = conn.cursor()

# 创建表
cur.execute("CREATE TABLE student(id integer,name varchar,sex varchar);")

# 插入数据
cur.execute("INSERT INTO student(id,name,sex)VALUES(%s,%s,%s)", (1, 'Aspirin', 'M'))
cur.execute("INSERT INTO student(id,name,sex)VALUES(%s,%s,%s)", (2, 'Taxol', 'F'))
cur.execute("INSERT INTO student(id,name,sex)VALUES(%s,%s,%s)", (3, 'Dixheral', 'M'))

# 获取结果
cur.execute('SELECT * FROM antennas')
results = cur.fetchall()
print(results)

# 关闭连接
conn.commit()
cur.close()
conn.close()