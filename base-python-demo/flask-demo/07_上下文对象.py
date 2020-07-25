from flask import request
from flask import Flask, session
from flask import abort, Response
from flask import make_response
from flask import jsonify
import json
from flask import g

app = Flask(__name__)
app.config["DEBUG"] = True


"""
    请求上下文
        request session
    应用上下文
        current_app 当前运行程序文件实例
        g   处理请求时，用于临时存储的对象，每次请求都会重设
        
"""

# g变量，相当于一个容器对象，可以用来存储变量
# 存储变量的方式和给对象添加属性的方式一样
g.name = "xiaoxuan"
# 每次进入视图函数之前都会把g变量清空掉，可以用来在视图函数中传递参数给其他函数


@app.route("/index", methods=(["GET", "POST"]))
def index():
    name = session.get("name")
    return name




if __name__ == "__main__":
    app.run()









