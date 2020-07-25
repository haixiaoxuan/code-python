from flask import request
from flask import Flask, session
from flask import abort, Response
from flask import make_response
from flask import jsonify
import json

app = Flask(__name__)
app.config["DEBUG"] = True

# session 密钥串
app.config["SECRET_KEY"] = "abcd..."


@app.route("/login", methods=(["GET", "POST"]))
def login():
    session["name"] = "xiaoxuan"
    session["age"] = 19
    return "success login"


@app.route("/index", methods=(["GET", "POST"]))
def index():
    name = session.get("name")
    return name




if __name__ == "__main__":
    app.run()









