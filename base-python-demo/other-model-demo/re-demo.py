import re

"""
    re.match # 只能从开头进行匹配
"""
# re.match
res = re.match(r"^x(\d{3})", "x123b456")
print(res.group())      # 匹配结果
print(res.group(1))     # 第一个小括号内匹配的内容
print(res.span())       # 匹配长度


# 贪婪匹配
res = re.match(".*(\d+)", "1234567")
print(res.group(1))     # 贪婪匹配
res = re.match(".*?(\d+)", "1234567")
print(res.group(1))     # 最小匹配


# 匹配换行符(. 本身不能匹配换行符)
res = re.match(".*(\d+)", "ab\n\n123", re.S)
print(res.group(1))


# re.search 找到符合条件的一个结果
res = re.search("\d+", "abc123d1234")
print(res.group())


# re.findall
res = re.findall("\d+", "abc123d1234")
print(res)


# re.sub  # 对匹配到的结果进行替换
res = re.sub("\d+", "", "abc123d456")
print(res)
# \1 表示匹配到字符
res = re.sub("(\d+)", r"\1 ---", "abc123d456")
print(res)