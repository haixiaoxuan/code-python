import optparse

"""
    @note:
        对命令行参数进行解析
    @author: xiexiaoxuan
    @e-mail: 281370705@qq.com
"""

import optparse

op = optparse.OptionParser()
op.add_option("-s", "--server", dest="server")
op.add_option("-p", "--port", dest="port")
values, args = op.parse_args()
print("values:", values)
print("args:", args)
print("values.server", values.server)
print(type(values))
