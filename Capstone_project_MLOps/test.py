import requests
import json

def test_flask_api():
    url = "http://localhost:8000/predict"  # Flask API endpoint

    data = {
        "cylinders": 4,
        "displacement": 140.0,
        "horsepower": 90,
        "weight": 2264,
        "acceleration": 15.5,
        "model_year": 76,
        "origin": "USA"
    }

    response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(data))

    if response.ok:
        print("Flask API Prediction:", response.json())
    else:
        print(f"Request failed with status {response.status_code}: {response.text}")

if __name__ == "__main__":
    test_flask_api()
