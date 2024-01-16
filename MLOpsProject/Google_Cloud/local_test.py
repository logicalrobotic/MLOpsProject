import requests
import json

resp = requests.post("http://127.0.0.1:5000", json=[1,2,3,4,5,6,7,8,9,10])

print(resp.json())