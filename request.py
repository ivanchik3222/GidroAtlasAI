import requests, json
url = 'http://localhost:5000/predict'
with open('payload.json', 'r', encoding='utf-8') as f:
    payload = json.load(f)
resp = requests.post(url, json=payload)
print(resp.status_code, resp.json())