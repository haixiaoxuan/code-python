import hashlib

"""
    @author: xiexiaoxuan
    @e-mail: 281370705@qq.com
"""


md_5 = hashlib.md5()
md_5.update("hello".encode("utf8"))
md_5.update("world".encode("utf8"))
data = md_5.hexdigest()
print(data)

md_52 = hashlib.md5("1".encode("utf8"))
md_52.update("helloworld".encode("utf8"))
data = md_52.hexdigest()
print(data)

md_52 = hashlib.md5("1".encode("utf8"))  # 加盐
md_52.update("helloworld".encode("utf8"))
data = md_52.hexdigest()
print(data)
''' 前两次输出结果一样，第三次加盐之后会发生改变 '''
