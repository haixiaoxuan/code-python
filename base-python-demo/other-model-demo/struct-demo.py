import struct

"""
    @author: xiexiaoxuan
    @e-mail: 281370705@qq.com
"""

res = struct.pack('i', 123)  # 将123 转为占四个字节的int
num = struct.unpack('i', res)
print(num)
print(res)