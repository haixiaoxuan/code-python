from flask import Flask
from flask import redirect, url_for
from werkzeug.routing import BaseConverter

"""
    app.url_map     查看整个app中的路由信息

"""

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route("/index", methods=["GET", "POST"])
def index():
    # 视图函数
    return "hello world"


# 重定向
@app.route("/redirect")
def redirect_test():
    url = url_for("index")      # 通过视图函数名称获取对应的url
    print(url)
    # return redirect(url)
    return redirect("http://www.baidu.com")


# 转换器
# @app.route("/transform/<int:num>")
@app.route("/transform/<abs>")      # 不加类型转换默认是string, other: int float path(接受斜线)
def transform_test(abs):
    return abs


# 自定义转换器
class MyTransform(BaseConverter):
    # 增加正则表达式类型
    def __init__(self, url_map, regex):
        super(MyTransform, self).__init__(url_map)
        self.regex = regex  # 此参数也可以在程序中写死
    def to_python(self, value):
        # 经过正则匹配后的字符串在经过次函数 然后传入视图函数
        return value
    def to_url(self, value):
        # 使用url_for方法的时候调用，先经过 to_url,在返回给 redirect
        # url_for("mytransform_test", num=100)
        return value


# 注册添加
app.url_map.converters["re"] = MyTransform
# 使用
@app.route("/mytransform/<re('\d+'):num>")
def mytransform_test(num):
    return num


if __name__ == "__main__":
    app.run()

