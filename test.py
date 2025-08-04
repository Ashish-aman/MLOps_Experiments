import requests

url = "http://localhost:8000/predict"
data = {"trip_distance": 3.2, "trip_duration": 720}
response = requests.post(url, json=data)
print(response.status_code)
print(response.text)
