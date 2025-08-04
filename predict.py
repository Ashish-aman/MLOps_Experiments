import pandas as pd
import pickle
import os

# Define a simple DataFrame as input
input_data = pd.DataFrame([{
    "trip_distance": 300,
    "trip_duration": 7.5
}])

# Path to model
model_path = "lr_model.bin"

# Check if model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f" Model file not found at: {model_path}")

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Run prediction
try:
    predictions = model.predict(input_data)
    print("Prediction output:", predictions.tolist())
except Exception as e:
    print("Error during prediction:", e)
    raise
