from dotenv import load_dotenv
load_dotenv()  # ✅ loads .env from project root

import os
import hopsworks
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# -----------------------------
# Read from .env (NO confusion)
# -----------------------------
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST")

if not HOPSWORKS_PROJECT or not HOPSWORKS_API_KEY or not HOPSWORKS_HOST:
    raise ValueError(
        "Missing Hopsworks env vars. Make sure your .env has:\n"
        "HOPSWORKS_PROJECT=...\n"
        "HOPSWORKS_API_KEY=...\n"
        "HOPSWORKS_HOST=...\n"
    )

FEATURE_GROUP_NAME = "aqi_features_v2"
FEATURE_GROUP_VERSION = 1

MODEL_REGISTRY_NAMES = {
    "RandomForest": "aqi_random_forest",
    "Ridge": "aqi_ridge",
    "NeuralNet": "aqi_neural_net",
}


# -----------------------------
# Helper: evaluation
# -----------------------------
def evaluate(name, y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"\n{name}")
    print("RMSE:", rmse)
    print("MAE :", mae)
    print("R2  :", r2)
    return rmse, mae, r2


# -----------------------------
# MAIN
# -----------------------------
print("✅ Logging in to Hopsworks...")
project = hopsworks.login(
    project=HOPSWORKS_PROJECT,
    api_key_value=HOPSWORKS_API_KEY,
    host=HOPSWORKS_HOST,
)
fs = project.get_feature_store()

print("✅ Reading Feature Group data...")
fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
df = fg.read()
print("Data loaded:", df.shape)

# -----------------------------
# Sort by timestamp
# -----------------------------
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
df = df.sort_values("timestamp_utc").reset_index(drop=True)

# -----------------------------
# ENGINEERED FEATURES (teacher requirement)
# -----------------------------
df["aqi_lag1"] = df["aqi"].shift(1)
df["aqi_lag24"] = df["aqi"].shift(24)
df["aqi_roll24"] = df["aqi"].rolling(24).mean()
df["aqi_roll7d"] = df["aqi"].rolling(24 * 7).mean()
df["aqi_diff1"] = df["aqi"] - df["aqi_lag1"]

# drop rows with NaN from engineered features
before = len(df)
df = df.dropna(subset=["aqi_lag1", "aqi_lag24", "aqi_roll24", "aqi_roll7d", "aqi_diff1"]).copy()
after = len(df)
print(f"Rows after engineered feature dropna: {after} (dropped {before - after})")

# -----------------------------
# FEATURES & TARGET
# -----------------------------
drop_cols = ["aqi", "timestamp_utc", "event_time", "city"]
X = df.drop(columns=drop_cols)
y = df["aqi"]

print("\n✅ Final training feature columns:")
print(list(X.columns))
print("X shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train 3 models
# -----------------------------
print("\n✅ Training RandomForest...")
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("✅ Training Ridge...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

print("✅ Training NeuralNet (MLPRegressor)...")
mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)

# -----------------------------
# Evaluate
# -----------------------------
rf_rmse, rf_mae, rf_r2 = evaluate("RandomForest", y_test, rf_pred)
ridge_rmse, ridge_mae, ridge_r2 = evaluate("Ridge", y_test, ridge_pred)
mlp_rmse, mlp_mae, mlp_r2 = evaluate("NeuralNet (MLP)", y_test, mlp_pred)

# -----------------------------
# Save local model files
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(ridge, "models/ridge.pkl")
joblib.dump(mlp, "models/neural_net.pkl")
print("\n✅ Models saved locally in /models")

# -----------------------------
# Save to Hopsworks Model Registry
# -----------------------------
print("\n✅ Saving models to Hopsworks Model Registry...")
mr = project.get_model_registry()

def register_model(model_name: str, model_path: str, metrics: dict):
    model = mr.python.create_model(
        name=model_name,
        metrics=metrics,
        description=f"{model_name} for Islamabad AQI forecasting (engineered features included)",
    )
    model.save(model_path)
    print(f"✅ Saved to Registry: {model_name}")

register_model(
    MODEL_REGISTRY_NAMES["RandomForest"],
    "models/random_forest.pkl",
    {"rmse": rf_rmse, "mae": rf_mae, "r2": rf_r2},
)

register_model(
    MODEL_REGISTRY_NAMES["Ridge"],
    "models/ridge.pkl",
    {"rmse": ridge_rmse, "mae": ridge_mae, "r2": ridge_r2},
)

register_model(
    MODEL_REGISTRY_NAMES["NeuralNet"],
    "models/neural_net.pkl",
    {"rmse": mlp_rmse, "mae": mlp_mae, "r2": mlp_r2},
)

print("\n🎉 DONE: Training + Registry upload completed!")
