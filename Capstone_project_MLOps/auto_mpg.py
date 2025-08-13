import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

# --------------------------------------
# Load and preprocess the Auto MPG dataset
# --------------------------------------
column_names = [
    'mpg', 'cylinders', 'displacement', 'horsepower',
    'weight', 'acceleration', 'model_year', 'origin', 'car_name'
]

df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data',
    sep='\s+',
    names=column_names
)

df['horsepower'] = df['horsepower'].replace('?', np.nan).astype(float)
df.dropna(inplace=True)
df.drop('car_name', axis=1, inplace=True)
df['origin'] = df['origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

X = df.drop('mpg', axis=1)
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
categorical_features = ['origin']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('encoder', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

print("Preprocessing complete.")
print("Processed training shape:", preprocessor.fit_transform(X_train).shape)

# --------------------------------------
# Model Training and Logging
# --------------------------------------
mlflow.set_tracking_uri("http://localhost:5555")
mlflow.set_experiment("Capstone_Model_Comparison")

models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR()
}

best_rmse = float('inf')
best_run_id = None
best_model = None

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        signature = infer_signature(X_test, preds)

        mlflow.log_param("model", name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=X_test.iloc[:5],
            signature=signature
        )

        print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

        if rmse < best_rmse:
            best_model = model
            best_rmse = rmse
            best_run_id = run.info.run_id

# --------------------------------------
# Save and Register Best Model
# --------------------------------------
best_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', best_model)
])
best_pipeline.fit(X_train, y_train)

with open("best_model_pipeline.bin", "wb") as f:
    pickle.dump(best_pipeline, f)

model_uri = f"runs:/{best_run_id}/model"
registered_model_name = "CapstoneAUTOMgp_BestModel"

mlflow.register_model(model_uri, registered_model_name)

print(f"Registered best model from run {best_run_id} with RMSE {best_rmse:.2f} as '{registered_model_name}'")
