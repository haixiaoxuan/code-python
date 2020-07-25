from flask import request
from flask import Flask
from flask import abort, Response
from flask import make_response
from flask import jsonify
import json

app = Flask(__name__)
app.config["DEBUG"] = True


# 设置cookies
@app.route("/set_cookies", methods=(["GET", "POST"]))
def index():
    response = make_response("success")
    # 其实set_cookie 就是设置 set-cookie
    response.set_cookie("name", "xiaoxuan")     # 默认是临时 cookies，浏览器关闭就失效
    response.set_cookie("age", "18", max_age=3600)
    # response.headers["Set-Cookie"] = "hh=xx; ii=zz;"
    return response


@app.route("/get_cookies", methods=(["GET", "POST"]))
def get_cookies():
    name = request.cookies.get("hh")
    return str(name)


@app.route("/del_cookies", methods=(["GET", "POST"]))
def del_cookies():
    response = make_response()
    response.delete_cookie("name")
    return response


if __name__ == "__main__":
    app.run()









