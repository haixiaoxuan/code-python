from bs4 import BeautifulSoup
import requests

"""
    @note:
        不需要编写正则完成对网页的提取
        "html.parser"   python内置，速度适中，容错强
        "lxml"          速度快，容错强
        "xml"           速度快，唯一支持xml的解析器
        "html5lib"      最好的容错性，速度慢
        
    @author: xiexiaoxuan
    @e-mail: 281370705@qq.com
"""

content = """
    <html>
        <head><title name="name">标题</title></head>
        <body>
    </html>
"""

soup = BeautifulSoup(content, "lxml")
print(soup.prettify())  # 对错误的html 进行容错
print(soup.title.string)    # 提取标题

# 标签选择器，如果有多个只返回一个结果
print(type(soup.title))
print(soup.head.title)
print(soup.title)
print(soup.title.name)

# 获取属性
print(soup.title["name"])
print(soup.title.attrs["name"])

# 拿到子节点以列表形式返回
print(soup.html.contents)
print(soup.html.children)
print(soup.html.descendants)    # 所有子孙标签
print(soup.title.parent)
print(soup.title.parents)

# 输出兄弟节点
print(soup.title.next_siblings)
print(soup.title.previous_siblings)


""" 标准选择器，返回单个元素 """
content = requests.get("http://www.baidu.com").text
soup = BeautifulSoup(content, "lxml")
print(soup.find_all("title"))


""" css 选择器 """
res = soup.select("#su")
print(res)


