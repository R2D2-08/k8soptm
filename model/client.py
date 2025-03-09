import requests
import json
import torch
response = requests.post("http://127.0.0.1:5000/predict", json={"input": torch.randn(10, 128).tolist()})
print("Server Response: ", json.loads(response.text)) if response.status_code == 200 else print("Error: ", response.status_code, response.text)
