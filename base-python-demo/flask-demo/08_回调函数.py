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
     这些 hook 是不区分请求的，如果想针对特定的请求，可以使用全局变量 request，
     通过request拿到请求path，然后通过path判断
     
     要想teardown_request 在视图函数发生异常时被执行，则需要 debug = False

"""


@app.before_first_request
def before1():
    """ 在第一次请求处理之前被执行 """
    print("before1")


@app.before_request
def before2():
    """ 在每一次请求处理之前被执行 """
    print("before2")


@app.after_request
def after(response):
    """ 在每次视图函数处理完之后都会被执行，前提是视图函数没有抛出异常 """
    print("after")
    return response


@app.teardown_request
def error(response):
    """ 在每次视图函数处理完之后都会被执行，无论是否出现异常 """
    print("error")
    return response


@app.route("/index", methods=(["GET", "POST"]))
def index():
    print("index")
    return "hello world"


if __name__ == "__main__":
    app.run()









