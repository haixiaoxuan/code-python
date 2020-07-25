import json
import numpy as np
import requests
from zh.model.utils import MNISTLoader


"""
    https://tf.wiki/zh/deployment/serving.html

    不考虑高并发和性能问题，其实配合 Flask 等 Python 下的 Web 框架就能非常轻松地实现服务器 API
    tensorFlow 为我们提供了 TensorFlow Serving 这一组件，能够帮助我们在实际生产环境中灵活且高性能地部署机器学习模型。

    TensorFlow Serving 可以使用 apt-get 或 Docker 安装。在生产环境中，推荐 使用 Docker 部署 TensorFlow Serving


    TensorFlow Serving 可以直接读取 SavedModel 格式的模型进行部署
        tensorflow_model_server \
            --rest_api_port=端口号（如8501） \
            --model_name=模型名 \
            --model_base_path="SavedModel格式模型的文件夹绝对地址（不含版本号）"


    TensorFlow Serving 支持热更新模型，其典型的模型文件夹结构如下：
    典型结构如下：
        /saved_model_files
            /1      # 版本号为1的模型文件
                /assets
                /variables
                saved_model.pb
            ...
            /N      # 版本号为N的模型文件
                /assets
                /variables
                saved_model.pb
    上面 1~N 的子文件夹代表不同版本号的模型。当指定 --model_base_path 时，只需要指定根目录的 绝对地址 （不是相对地址）即可。
    例如，如果上述文件夹结构存放在 home/snowkylin 文件夹内，则 --model_base_path 应当设置为 home/snowkylin/saved_model_files （不附带模型版本号）。
    TensorFlow Serving 会自动选择版本号最大的模型进行载入。


    对于自定义的keras_model,对格式要求较高
        需要导出到 SavedModel 格式的方法（比如 call ）不仅需要使用 @tf.function 修饰，还要在修饰时指定 input_signature 参数，
        以显式说明输入的形状。该参数传入一个由 tf.TensorSpec 组成的列表，指定每个输入张量的形状和类型。
        例如，对于 MNIST 手写体数字识别，我们的输入是一个 [None, 28, 28, 1] 的四维张量（ None 表示第一维即 Batch Size 的大小不固定），
        此时我们可以将模型的 call 方法做以下修饰
            @tf.function(input_signature=[tf.TensorSpec([None, 28, 28, 1], tf.float32)])
            def call(self, inputs):
        在将模型使用 tf.saved_model.save 导出时，需要通过 signature 参数提供待导出的函数的签名（Signature）。
        简单说来，由于自定义的模型类里可能有多个方法都需要导出，因此，需要告诉 TensorFlow Serving 每个方法在被客户端调用时分别叫做什么名字。
        例如，如果我们希望客户端在调用模型时使用 call 这一签名来调用 model.call 方法时，我们可以在导出时传入 signature 参数，
        以 dict 的键值对形式告知导出的方法对应的签名，代码如下：
            tf.saved_model.save(model, "saved_with_signature/1", signatures={"call": model.call})

"""


"""
    在客户端调用以 TensorFlow Serving 部署的模型 
    TensorFlow Serving 支持以 gRPC 和 RESTful API 调用以 TensorFlow Serving 部署的模型。
    RESTful API 以标准的 HTTP POST 方法进行交互，请求和回复均为 JSON 对象。为了调用服务器端的模型，我们在客户端向服务器发送以下格式的请求
        
        服务器 URI： http://服务器地址:端口号/v1/models/模型名:predict
        
        请求内容：
        {
            "signature_name": "需要调用的函数签名（Sequential模式不需要）",
            "instances": 输入数据
        }
        回复：
        {
            "predictions": 返回值
        }

"""


data_loader = MNISTLoader()
data = json.dumps({
    "instances": data_loader.test_data[0:3].tolist()
    })
headers = {"content-type": "application/json"}
json_response = requests.post(
    'http://localhost:8501/v1/models/MLP:predict',
    data=data, headers=headers)
predictions = np.array(json.loads(json_response.text)['predictions'])
print(np.argmax(predictions, axis=-1))
print(data_loader.test_label[0:10])





