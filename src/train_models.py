# src/train_models.py

import hopsworks
import pandas as pd
import numpy as np
from config import HOPSWORKS_PROJECT, HOPSWORKS_API_KEY, HOPSWORKS_HOST

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# ---------------- LOGIN ----------------
project = hopsworks.login(
    project=HOPSWORKS_PROJECT,
    api_key_value=HOPSWORKS_API_KEY,
    host=HOPSWORKS_HOST,
)
fs = project.get_feature_store()

# ---------------- LOAD DATA ----------------
fg = fs.get_feature_group("aqi_features_v2", version=1)
df = fg.read()
print("Data loaded:", df.shape)

# ---------------- FEATURES & TARGET ----------------
X = df.drop(columns=["aqi", "timestamp_utc", "event_time", "city"])
y = df["aqi"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- MODEL 1 Random Forest ----------------
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# ---------------- MODEL 2 Ridge ----------------
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

# ---------------- MODEL 3 Simple Neural Network ----------------
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(64,64), max_iter=500)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)

# ---------------- EVALUATION ----------------
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name}")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)
    return rmse

rf_rmse = evaluate("Random Forest", y_test, rf_pred)
ridge_rmse = evaluate("Ridge Regression", y_test, ridge_pred)
mlp_rmse = evaluate("Neural Network", y_test, mlp_pred)

# ---------------- SAVE MODELS ----------------
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(ridge, "models/ridge.pkl")
joblib.dump(mlp, "models/neural_net.pkl")

print("\nModels saved in /models folder")

# ---------------- SAVE TO HOPSWORKS MODEL REGISTRY ----------------
mr = project.get_model_registry()

def register_model(model_name: str, model_path: str, metrics: dict):
    model = mr.python.create_model(
        name=model_name,
        metrics=metrics,
        description=f"{model_name} for Islamabad AQI forecasting"
    )
    model.save(model_path)
    print(f"✅ Saved to Registry: {model_name}")

register_model(
    "aqi_random_forest",
    "models/random_forest.pkl",
    {"rmse": float(rf_rmse), "mae": float(mean_absolute_error(y_test, rf_pred)), "r2": float(r2_score(y_test, rf_pred))}
)

register_model(
    "aqi_ridge",
    "models/ridge.pkl",
    {"rmse": float(ridge_rmse), "mae": float(mean_absolute_error(y_test, ridge_pred)), "r2": float(r2_score(y_test, ridge_pred))}
)

register_model(
    "aqi_neural_net",
    "models/neural_net.pkl",
    {"rmse": float(mlp_rmse), "mae": float(mean_absolute_error(y_test, mlp_pred)), "r2": float(r2_score(y_test, mlp_pred))}
)
