from celery import Celery

"""
    @note:
        创建 celery实例
    @author: xiexiaoxuan
    @e-mail: 281370705@qq.com
"""


app = Celery('proj', include=['proj.tasks'])

# 配置
app.config_from_object('proj.config')


if __name__ == "__main__":

    app.start()