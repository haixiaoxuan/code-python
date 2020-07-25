from flask import Flask


# 创建flask应用对象
# flask以__name__这个文件所在的目录为总目录，默认这个目录中的static为静态目录，templates为模板目录
app = Flask(__name__,
            static_url_path="/python",  # 访问静态资源的url前缀
            static_folder="static",
            template_folder="templates")

# DEBUG模式启动可以进行热加载， ctrl + s
app.config["DEBUG"] = True


# 添加配置的三种方式
# app.config.from_pyfile("conf.cfg")      # 使用配置文件
# class Config(object):
#     DEBUG=True
# app.config.from_object(Config)          # 使用配置对象
# app.config["DEBUG"] = True              # 直接操作字典对象


# 获取配置, 两种方式
# app.config.get("key")
# from flask import current_app
# current_app.config.get("key")


@app.route("/")
def index():
    return "hello world"


if __name__ == "__main__":
    # app.run()
    app.run(host="0.0.0.0", port="5000")    # 允许所有主机访问

