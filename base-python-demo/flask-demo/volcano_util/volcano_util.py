from flask import Flask
from flask import request
import json
from flask import make_response


app = Flask(__name__)
app.config["DEBUG"] = True

PATH = "/opt/volcano_file/"

def get_response(code, content):
    response = make_response(content)
    response.status = code
    return response


@app.route("/")
def index():
    """ 测试 """
    return "hello world"


@app.route("/upload", methods=(["POST"]))
def upload():
    """ 上传文件 """
    file = request.files.get("data")
    if file is None:
        return get_response("405", "file is null")

    content = request.form.get("json")
    if content is None:
        return get_response("405", "filename is null")

    content = json.loads(content)
    print("content ==> {0}".format(content))

    filename = content.get("filename")
    if filename is None:
        return get_response("405", "filename is null")
    if (not filename.endswith("xml")) and (not filename.endswith("dxf")):
        return get_response("405", "only support xml|dxf")
    file.save(PATH + filename)

    return get_response("200", "success")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")



