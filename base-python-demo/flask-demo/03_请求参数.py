from flask import request
from flask import Flask

app = Flask(__name__)
app.config["DEBUG"] = True



@app.route("/test", methods=(["GET", "POST"]))
def index():
    # 记录请求数据
    data = request.data
    print("data ", data)

    # 接受查询串中的参数 key=value&key2=value2
    args = request.args
    print("args ", args)

    # 表单数据
    name = request.form.get("name")
    names = request.form.getlist("name")
    print(request.data)

    # cookies信息
    cookies = request.cookies
    print("cookies ", cookies)

    # headers
    headers = request.headers
    print("headers ", headers)

    # 请求方式 get post
    method = request.method
    print("method ", method)

    # url
    url = request.url
    print("url ", url)

    files = request.files
    print("file ", files)

    # 可以获取json格式数据 要求前端发送过来的数据为Content-Type : application/json
    json = request.get_json()
    print(json)

    return "hello world"


# 上传文件
@app.route("/upload", methods=(["GET", "POST"]))
def upload():

    file = request.files.get("data")
    if file is None:
        return "文件为空"

    read = file.read()
    print(read)

    # file.save("./aa.json")
    return "上传成功"


if __name__ == "__main__":
    app.run()









