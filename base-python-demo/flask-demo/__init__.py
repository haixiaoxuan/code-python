import requests
import json


url = "http://localhost:5000/"
# requests.get(url + "test?name=xiaoxuan")


files = {
    'json': (None, json.dumps({"name": "xiaoxuan"}), 'application/json'),
    # 'data': (open(r'01_hello-world.py', 'rb'))
}
r = requests.post(url + 'test', files=files)
