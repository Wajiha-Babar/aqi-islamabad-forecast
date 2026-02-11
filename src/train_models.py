from dotenv import load_dotenv
load_dotenv()  # loads .env from project root

import os
import json
import time
from datetime import datetime, timezone

import hopsworks
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# -----------------------------
# ENV (from .env)
# -----------------------------
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST")

if not HOPSWORKS_PROJECT or not HOPSWORKS_API_KEY or not HOPSWORKS_HOST:
    raise ValueError(
        "Missing Hopsworks env vars. Ensure .env has:\n"
        "HOPSWORKS_PROJECT=...\n"
        "HOPSWORKS_API_KEY=...\n"
        "HOPSWORKS_HOST=...\n"
    )

# -----------------------------
# SETTINGS (same)
# -----------------------------
FEATURE_GROUP_NAME = "aqi_features_v2"
FEATURE_GROUP_VERSION = 1

MODEL_REGISTRY_NAMES = {
    "RandomForest": "aqi_random_forest",
    "Ridge": "aqi_ridge",
    "NeuralNet": "aqi_neural_net",
    "Best": "aqi_best_model",
}

TRAIN_DAYS = 180
TRAIN_MAX_ROWS = 8000
TEST_RATIO = 0.20

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# -----------------------------
# Helpers
# -----------------------------
def evaluate(name: str, y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"\n{name}")
    print("RMSE:", rmse)
    print("MAE :", mae)
    print("R2  :", r2)
    return {"rmse": rmse, "mae": mae, "r2": r2}

def time_split(X, y, test_ratio: float):
    split_idx = int(len(X) * (1 - test_ratio))
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

def print_ridge_formula(ridge_pipeline: Pipeline, feature_names: list, top_k: int = 12):
    model = ridge_pipeline.named_steps["model"]
    coefs = model.coef_
    intercept = model.intercept_

    pairs = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    print("\n📌 Ridge (formula-based) learned equation (top terms):")
    print(f"Intercept: {intercept:.6f}")
    for name, c in pairs:
        print(f"  {name:>15}: {c:.6f}")

# ✅ UPDATED: embed rmse in description so inference can ALWAYS read it reliably
def register_model(mr, model_name: str, model_path: str, metrics: dict, description: str):
    rmse = metrics.get("rmse", None)
    desc = f"{description} | rmse={rmse}"  # ✅ stable

    model = mr.python.create_model(
        name=model_name,
        metrics=metrics,          # keep as-is
        description=desc,         # ✅ updated
    )
    model.save(model_path)
    print(f"✅ Saved to Registry: {model_name}")


# -----------------------------
# MAIN
# -----------------------------
t0 = time.time()
print("✅ Logging in to Hopsworks...")
project = hopsworks.login(
    project=HOPSWORKS_PROJECT,
    api_key_value=HOPSWORKS_API_KEY,
    host=HOPSWORKS_HOST,
)
fs = project.get_feature_store()
mr = project.get_model_registry()
print("✅ Logged in.")

print("✅ Reading Feature Group data...")
fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
df = fg.read()
print("Data loaded:", df.shape)

df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.floor("h")
df = df.sort_values("timestamp_utc").reset_index(drop=True)

cutoff = df["timestamp_utc"].max() - pd.Timedelta(days=TRAIN_DAYS)
df = df[df["timestamp_utc"] >= cutoff].copy()

if len(df) > TRAIN_MAX_ROWS:
    df = df.iloc[-TRAIN_MAX_ROWS:].copy()

print(f"✅ Using last {TRAIN_DAYS} days. Rows now: {len(df)}")

# -----------------------------
# ENGINEERED FEATURES (no leakage)
# -----------------------------
df["aqi_lag1"] = df["aqi"].shift(1)
df["aqi_lag2"] = df["aqi"].shift(2)
df["aqi_lag24"] = df["aqi"].shift(24)

df["aqi_roll24"] = df["aqi"].shift(1).rolling(24).mean()
df["aqi_roll7d"] = df["aqi"].shift(1).rolling(24 * 7).mean()

df["aqi_diff1"] = df["aqi_lag1"] - df["aqi_lag2"]

before = len(df)
df = df.dropna(subset=["aqi_lag1", "aqi_lag2", "aqi_lag24", "aqi_roll24", "aqi_roll7d", "aqi_diff1", "aqi"]).copy()
after = len(df)
print(f"Rows after engineered feature dropna: {after} (dropped {before - after})")

# -----------------------------
# FEATURES & TARGET
# -----------------------------
drop_cols = ["aqi", "timestamp_utc", "event_time", "city"]
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols)
y = df["aqi"].astype(float)

print("\n✅ Final training feature columns:")
print(list(X.columns))
print("X shape:", X.shape)

X_train, X_test, y_train, y_test = time_split(X, y, TEST_RATIO)
print("Train:", X_train.shape, "Test:", X_test.shape)

# -----------------------------
# Train 3 models
# -----------------------------
print("\n✅ Training RandomForest...")
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_depth=14,
    min_samples_leaf=2,
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("✅ Training Ridge (formula-based)...")
ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0)),
])
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

print("✅ Training NeuralNet (MLPRegressor)...")
mlp = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MLPRegressor(
        hidden_layer_sizes=(64, 64),
        max_iter=400,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=10,
    )),
])
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)

# -----------------------------
# Evaluate
# -----------------------------
rf_metrics = evaluate("RandomForest", y_test, rf_pred)
ridge_metrics = evaluate("Ridge", y_test, ridge_pred)
mlp_metrics = evaluate("NeuralNet (MLP)", y_test, mlp_pred)

print_ridge_formula(ridge, list(X.columns), top_k=12)

# Pick best
all_metrics = {
    "RandomForest": rf_metrics,
    "Ridge": ridge_metrics,
    "NeuralNet": mlp_metrics,
}
best_name = min(all_metrics.keys(), key=lambda k: all_metrics[k]["rmse"])
best_metrics = all_metrics[best_name]
print(f"\n🏆 BEST MODEL: {best_name}  -> {best_metrics}")

# Save local
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(ridge, "models/ridge.pkl")
joblib.dump(mlp, "models/neural_net.pkl")

best_path = "models/best_model.pkl"
joblib.dump({"name": best_name, "model": {"RandomForest": rf, "Ridge": ridge, "NeuralNet": mlp}[best_name]}, best_path)
print("\n✅ Models saved locally in /models")

# Save metrics proof
os.makedirs("outputs", exist_ok=True)
metrics_path = f"outputs/metrics_{RUN_ID}.json"
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump({
        "run_id": RUN_ID,
        "train_days": TRAIN_DAYS,
        "train_max_rows": TRAIN_MAX_ROWS,
        "test_ratio": TEST_RATIO,
        "feature_cols": list(X.columns),
        "metrics": all_metrics,
        "best_model": best_name,
        "best_metrics": best_metrics,
    }, f, indent=2)
print("✅ Saved metrics:", metrics_path)

# Upload to Model Registry
print("\n✅ Saving models to Hopsworks Model Registry...")

register_model(
    mr,
    MODEL_REGISTRY_NAMES["RandomForest"],
    "models/random_forest.pkl",
    rf_metrics,
    description=f"{MODEL_REGISTRY_NAMES['RandomForest']} (6 months, time-split, engineered features). Run={RUN_ID}",
)

register_model(
    mr,
    MODEL_REGISTRY_NAMES["Ridge"],
    "models/ridge.pkl",
    ridge_metrics,
    description=f"{MODEL_REGISTRY_NAMES['Ridge']} FORMULA-based (Ridge) (6 months, time-split). Run={RUN_ID}",
)

register_model(
    mr,
    MODEL_REGISTRY_NAMES["NeuralNet"],
    "models/neural_net.pkl",
    mlp_metrics,
    description=f"{MODEL_REGISTRY_NAMES['NeuralNet']} (MLP) (6 months, time-split). Run={RUN_ID}",
)

register_model(
    mr,
    MODEL_REGISTRY_NAMES["Best"],
    best_path,
    best_metrics,
    description=f"{MODEL_REGISTRY_NAMES['Best']} BEST={best_name} (6 months, time-split). Run={RUN_ID}",
)

print("\n🎉 DONE: Training + Registry upload completed!")
print("⏱️ Total time:", f"{time.time() - t0:.2f}s")
