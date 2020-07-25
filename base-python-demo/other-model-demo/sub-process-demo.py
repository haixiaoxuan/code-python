import subprocess

"""
    @author: xiexiaoxuan
    @e-mail: 281370705@qq.com
"""

res = subprocess.Popen("dir", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# 将结果输进管道
msg = res.stdout.read()
# 从管道中进行读取
print(msg.decode("gbk"))
