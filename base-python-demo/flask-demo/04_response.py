from flask import request
from flask import Flask
from flask import abort, Response
from flask import make_response
from flask import jsonify
import json

app = Flask(__name__)
app.config["DEBUG"] = True



# 异常终止
# @app.route("/test", methods=(["GET", "POST"]))
def index():

    # abort 函数可以立即终止函数执行
    # 1. 必须是标准的 http 状态码
    # 2. 传递响应体信息
    abort(404)
    response = Response("异常")
    abort(response)
    return "hello world"


# 设置response
# @app.route("/test", methods=(["GET", "POST"]))
def index():
    response = make_response("eeeeeee")
    response.status = "999 itcast"      # 设置状态码
    response.headers["city"] = "hh"     # 设置响应头
    return response


# 返回 json 串
@app.route("/test", methods=(["GET", "POST"]))
def index():
    data = {"name": "xiaoxuan", "age": 18}

    # return json.dumps(data), 200, {"Content-type": "application/json"}
    return jsonify(data)    # jsonify 相当于帮我们组装返回值


if __name__ == "__main__":
    app.run()









