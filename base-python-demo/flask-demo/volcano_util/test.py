import requests
import json


files = {
    'json': (None, json.dumps({"filename": "aa.dxf"}), 'application/json'),
    'data': (open('volcano_util.py', 'rb'))
}

host = "http://172.30.4.106:5000/"

r = requests.post(host + "upload", files=files)
print(r.status_code)
print(r.content.decode("utf8"))








