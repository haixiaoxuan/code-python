#! -*-encoding=utf8-*-

import pymysql
import time
import threading

# 简易版数据库连接池
def get_pool():
    l=[]
    for i in range(100):
        db = pymysql.connect(host="172.30.4.129",
                             port=3306,
                             user="root",
                             passwd="abcd.1234",
                             db="airflow")
        l.append(db)
    return l

g = (j for j in range(3000000, 5000000))
print(g)

def insert(sql,pool):
    conn = pool.pop()
    cursor = conn.cursor()
    try :
        while True:
            index=next(g)
            print(index,threading.current_thread().getName())
            cursor.execute(sql.format(index))
    except Exception as e:
        conn.commit()
        pool.append(conn)
        print(" error "+threading.current_thread().getName())



"""
    事务：
    conn.autocommit(False)  # 关闭自动提交事务
    conn.begin()    # 开启事务
    打开游标....
    conn.commit()
    
    :except
        conn.rollback()
"""




if __name__=="__main__":
    sql = "insert into student values ('xiaoxuan{0}',{0})"
    pool = get_pool()
    start=time.time()

    for i in range(30):
        t=threading.Thread(target=insert,args=(sql,pool))
        t.start()
        # t.join()
    end=time.time()
    print("over")
    print(end-start)





















