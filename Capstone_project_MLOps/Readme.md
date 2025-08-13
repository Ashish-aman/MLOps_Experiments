
# Fuel Efficiency Prediction with MLflow

## Overview
VM Public ip - 52.147.206.68
.pem is in repo
This project implements a complete MLflow pipeline for predicting vehicle fuel efficiency (MPG) using the Auto MPG dataset. It demonstrates:

- Model training and comparison with MLflow tracking  
- Model versioning and rollback with MLflow Model Registry  
- Deployment of the best model as a Flask API in Docker  
- Monitoring data and prediction drift with Evidently AI dashboards  

---

## Project Structure

```

├── predict.py              # Flask app to serve model predictions
├── best_model_pipeline.bin # Saved model pipeline
├── Dockerfile              # Docker configuration for Flask app
├── test.py                 # Script to test Flask API endpoint
├── data_drift_report.html  # Sample data drift monitoring report
├── mlflow\_runs/            # MLflow experiment tracking directory
├── requirements.txt        # Python dependencies
└── README.md               # This file

````

---

## Setup Instructions

1. **Clone the repository:**

```bash
git clone <repo-url>
cd <repo-folder>
````

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run MLflow UI to monitor experiments:**

```bash
mlflow ui
```

Open `http://localhost:5500` in your browser.

---

## Model Training and Experiment Tracking

* Train your model and log experiments using MLflow in your training script.
* Register the best model to the MLflow Model Registry.
* Use the MLflow UI to compare models, view metrics, and manage model versions.

---

## Deployment with Flask and Docker

1. **Build the Docker image:**


docker build -t mpg-flask-app .


2. **Run the Docker container:**


docker run -p 8000:8000 mpg-flask-app


3. **Test the API endpoint:**


python test.py


Or using curl:


curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"cylinders":4,"displacement":140.0,"horsepower":90,"weight":2264,"acceleration":15.5,"model_year":76,"origin":"USA"}'


---

## Monitoring and Data Drift Detection

* Use Evidently AI to generate data drift reports comparing live data with training data.
* Drift reports are saved as HTML files (`data_drift_report.html`) and can be viewed locally.
* Set up alerts to detect significant drift and trigger model retraining or review.

---

## Notes

* The model was trained using scikit-learn 1.6.1; using compatible versions is recommended to avoid unpickle errors.
* The Flask app serves predictions on port 8000 by default.
* For production deployment, consider using a WSGI server like Gunicorn instead of Flask’s development server.

---

## Contact

For questions or support, contact Ashish Aman - aaa534436@tatamotors.com

---




