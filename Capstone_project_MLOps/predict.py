import pickle
from flask import Flask, request, jsonify
import pandas as pd

# Load the saved model
with open("best_model_pipeline.bin", "rb") as f_in:
    model = pickle.load(f_in)

app = Flask("mpg-predictor")

# Define how to prepare features from input JSON
def prepare_features(car):
    return pd.DataFrame([{
        'cylinders': car['cylinders'],
        'displacement': car['displacement'],
        'horsepower': car['horsepower'],
        'weight': car['weight'],
        'acceleration': car['acceleration'],
        'model_year': car['model_year'],
        'origin': car['origin']  # Must be 'USA', 'Europe', or 'Japan'
    }])

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    car = request.get_json()
    features = prepare_features(car)
    pred = model.predict(features)[0]

    return jsonify({'predicted_mpg': round(float(pred), 2)})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)
