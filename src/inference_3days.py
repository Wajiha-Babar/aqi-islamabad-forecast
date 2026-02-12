from dotenv import load_dotenv
load_dotenv()

import os
import re
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import pandas as pd
import requests
import joblib
import hopsworks
MODEL_CACHE_DIR = Path("model_cache")
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# -----------------------------
# SETTINGS
# -----------------------------
FEATURE_GROUP_NAME = "aqi_features_v2"
FEATURE_GROUP_VERSION = 1

PRED_FG_NAME = "aqi_predictions_3days"
PRED_FG_VERSION = 4

CITY = "Islamabad"
LAT, LON = 33.6844, 73.0479

HAZARDOUS_THRESHOLD = 200

MODEL_NAMES = {
    "RandomForest": "aqi_random_forest",
    "Ridge": "aqi_ridge",
    "NeuralNet": "aqi_neural_net",
}

EXPECTED_FEATURES = [
    "hour", "day", "month", "day_of_week",
    "temp_c", "humidity", "pressure", "wind_speed",
    "aqi_lag1", "aqi_lag2", "aqi_lag24",
    "aqi_roll24", "aqi_roll7d",
    "aqi_diff1",
]

OUTPUT_DIR = Path("outputs")
OUTPUT_CSV = OUTPUT_DIR / "predictions_3days.csv"


# -----------------------------
# ENV
# -----------------------------
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST")

if not HOPSWORKS_PROJECT or not HOPSWORKS_API_KEY or not HOPSWORKS_HOST:
    raise ValueError("Missing Hopsworks env vars in .env")


# -----------------------------
# Helpers
# -----------------------------
def ensure_outputs_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def next_72_hours_utc() -> pd.DatetimeIndex:
    now = datetime.now(timezone.utc)
    start = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return pd.date_range(start=start, periods=72, freq="h", tz="UTC")

def fetch_open_meteo_forecast_utc(lat: float, lon: float) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,windspeed_10m",
        "forecast_days": 7,
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()

    h = j.get("hourly") or {}
    times = h.get("time") or []
    if not times:
        raise RuntimeError("Open-Meteo returned no hourly times.")

    ts = pd.to_datetime(times, utc=True).floor("h")
    df = pd.DataFrame({
        "timestamp_utc": ts,
        "temp_c": h.get("temperature_2m"),
        "humidity": h.get("relative_humidity_2m"),
        "pressure": h.get("surface_pressure"),
        "wind_speed": h.get("windspeed_10m"),
    })
    return df.dropna().drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc")

def extract_rmse(model_meta) -> float:
    for attr in ["training_metrics", "metrics"]:
        m = getattr(model_meta, attr, None)
        if isinstance(m, dict) and m:
            if "rmse" in m:
                return float(m["rmse"])
            for k, v in m.items():
                if "rmse" in str(k).lower():
                    return float(v)

    desc = getattr(model_meta, "description", "") or ""
    match = re.search(r"rmse\s*=\s*([0-9]*\.?[0-9]+)", desc, re.IGNORECASE)
    if match:
        return float(match.group(1))

    return 1e18

# def download_and_load_latest_model(mr, model_name: str):
#     versions = mr.get_models(model_name)
#     if not versions:
#         raise RuntimeError(f"No versions found for model '{model_name}'")

#     latest = sorted(versions, key=lambda x: x.version)[-1]
#     meta = mr.get_model(model_name, version=latest.version)

#     model_dir = Path(meta.download())
#     candidates = list(model_dir.rglob("*.pkl")) + list(model_dir.rglob("*.joblib"))
#     if not candidates:
#         raise RuntimeError(f"Downloaded model but no .pkl/.joblib found in: {model_dir}")

#     model_obj = joblib.load(candidates[0])
#     rmse = extract_rmse(meta)
#     return model_obj, float(rmse), int(latest.version)
def download_and_load_latest_model(mr, model_name: str):
    versions = mr.get_models(model_name)
    if not versions:
        raise RuntimeError(f"No versions found for model '{model_name}'")

    latest = sorted(versions, key=lambda x: x.version)[-1]
    ver = int(latest.version)

    # ✅ local cache path
    local_path = MODEL_CACHE_DIR / f"{model_name}_v{ver}.joblib"

    # ✅ if already cached -> load instantly
    if local_path.exists():
        model_obj = joblib.load(local_path)
        meta = mr.get_model(model_name, version=ver)
        rmse = extract_rmse(meta)
        return model_obj, float(rmse), ver

    # ✅ else download once
    meta = mr.get_model(model_name, version=ver)
    model_dir = Path(meta.download())
    candidates = list(model_dir.rglob("*.pkl")) + list(model_dir.rglob("*.joblib"))
    if not candidates:
        raise RuntimeError(f"Downloaded model but no .pkl/.joblib found in: {model_dir}")

    model_obj = joblib.load(candidates[0])

    # ✅ save to cache for next runs
    joblib.dump(model_obj, local_path)

    rmse = extract_rmse(meta)
    return model_obj, float(rmse), ver

def load_history_from_feature_store(fs, days: int = 40) -> pd.DataFrame:
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.floor("h")
    df = df.sort_values("timestamp_utc").dropna(subset=["aqi"]).copy()

    end = df["timestamp_utc"].max()
    start = end - pd.Timedelta(days=days)
    df = df[df["timestamp_utc"] >= start].copy()

    if df.empty:
        raise RuntimeError("History is empty.")

    df = (
        df[["timestamp_utc", "aqi"]]
        .drop_duplicates(subset=["timestamp_utc"])
        .set_index("timestamp_utc")
        .sort_index()
    )

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    df = df.reindex(full_idx)
    df["aqi"] = df["aqi"].ffill().bfill()

    return df.reset_index().rename(columns={"index": "timestamp_utc"})

def compute_engineered_features(history_aqi: pd.Series) -> Dict[str, float]:
    aqi_lag1 = float(history_aqi.iloc[-1])
    aqi_lag2 = float(history_aqi.iloc[-2]) if len(history_aqi) >= 2 else aqi_lag1
    aqi_lag24 = float(history_aqi.iloc[-24]) if len(history_aqi) >= 24 else float(history_aqi.iloc[0])

    last24 = history_aqi.iloc[-24:] if len(history_aqi) >= 24 else history_aqi
    last7d = history_aqi.iloc[-168:] if len(history_aqi) >= 168 else history_aqi

    aqi_roll24 = float(last24.mean())
    aqi_roll7d = float(last7d.mean())
    aqi_diff1 = float(aqi_lag1 - aqi_lag2)

    return {
        "aqi_lag1": aqi_lag1,
        "aqi_lag2": aqi_lag2,
        "aqi_lag24": aqi_lag24,
        "aqi_roll24": aqi_roll24,
        "aqi_roll7d": aqi_roll7d,
        "aqi_diff1": aqi_diff1,
    }

def save_predictions_to_feature_store(fs, df_preds: pd.DataFrame):
    fg = fs.get_or_create_feature_group(
        name=PRED_FG_NAME,
        version=PRED_FG_VERSION,
        primary_key=["city", "event_time"],
        description="Next 3 days AQI predictions (3 models + best + RMSEs + alerts)",
        online_enabled=True,
    )

    df_preds = df_preds.copy()

    if "event_time" not in df_preds.columns:
        df_preds["event_time"] = pd.to_datetime(df_preds["timestamp_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    df_preds["timestamp_utc"] = pd.to_datetime(df_preds["timestamp_utc"], utc=True)
    df_preds["hazardous_alert"] = df_preds["hazardous_alert"].astype(bool)

    df_preds.columns = [c.lower() for c in df_preds.columns]

    print("✅ Writing predictions to Feature Store (upsert)...")
    fg.insert(df_preds, write_options={"wait_for_job": False, "upsert": True})
    print("✅ Predictions stored (online).")


# -----------------------------
# MAIN
# -----------------------------
def run_inference_3days() -> Tuple[pd.DataFrame, str, float]:
    ensure_outputs_dir()

    print("✅ Step 1: Login to Hopsworks...")
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY,
        host=HOPSWORKS_HOST,
    )
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    print("✅ Step 2: Load AQI history (seed)...")
    hist_df = load_history_from_feature_store(fs, days=40)
    history_aqi = hist_df["aqi"].astype(float).copy()
    print(f"   History points: {len(history_aqi)}")

    print("✅ Step 3: Load 3 models from Model Registry...")
    models, rmses, versions = {}, {}, {}
    for pretty, reg in MODEL_NAMES.items():
        model_obj, rmse, ver = download_and_load_latest_model(mr, reg)
        models[pretty] = model_obj
        rmses[pretty] = float(rmse)
        versions[pretty] = int(ver)
        print(f"   -> {pretty}: v{ver} rmse={rmse}")

    # ✅ NEW: store individual RMSEs
    rmse_rf = float(rmses["RandomForest"])
    rmse_ridge = float(rmses["Ridge"])
    rmse_mlp = float(rmses["NeuralNet"])

    print("✅ Step 4: Choose BEST model...")
    best_model = min(rmses.keys(), key=lambda k: rmses[k])
    best_rmse = float(rmses[best_model])
    best_reason = f"Selected because it has the lowest RMSE ({best_rmse:.6f}) among all models."
    print(f"🏆 BEST: {best_model} | RMSE={best_rmse:.6f}")

    print("✅ Step 5: Fetch weather forecast for next 72 hours...")
    weather_df = fetch_open_meteo_forecast_utc(LAT, LON)
    future_times = next_72_hours_utc()

    weather_72 = (
        weather_df.set_index("timestamp_utc")
        .reindex(future_times)
        .ffill()
        .bfill()
        .reset_index()
        .rename(columns={"index": "timestamp_utc"})
    )

    print("✅ Step 6: Predict next 72 hours...")
    rows: List[Dict] = []

    for i, ts in enumerate(future_times):
        w = weather_72.iloc[i]

        base_features = {
            "hour": int(ts.hour),
            "day": int(ts.day),
            "month": int(ts.month),
            "day_of_week": int(ts.dayofweek),
            "temp_c": float(w["temp_c"]),
            "humidity": float(w["humidity"]),
            "pressure": float(w["pressure"]),
            "wind_speed": float(w["wind_speed"]),
        }

        eng = compute_engineered_features(history_aqi)

        X_row = pd.DataFrame([{**base_features, **eng}])
        X_row = X_row.reindex(columns=EXPECTED_FEATURES, fill_value=0.0)

        pred_rf = float(models["RandomForest"].predict(X_row)[0])
        pred_ridge = float(models["Ridge"].predict(X_row)[0])
        pred_mlp = float(models["NeuralNet"].predict(X_row)[0])

        best_pred = {"RandomForest": pred_rf, "Ridge": pred_ridge, "NeuralNet": pred_mlp}[best_model]
        history_aqi = pd.concat([history_aqi, pd.Series([best_pred])], ignore_index=True)

        rows.append({
            "timestamp_utc": ts,
            "city": CITY,
            **base_features,
            **eng,

            "pred_randomforest": pred_rf,
            "pred_ridge": pred_ridge,
            "pred_neuralnet": pred_mlp,

            # ✅ BEST INFO
            "best_model": best_model,
            "best_model_rmse": best_rmse,
            "best_reason": best_reason,
            "best_pred": best_pred,

            # ✅ NEW: individual RMSE columns (same for all 72 rows)
            "rmse_randomforest": rmse_rf,
            "rmse_ridge": rmse_ridge,
            "rmse_neuralnet": rmse_mlp,

            "hazardous_alert": bool(best_pred >= HAZARDOUS_THRESHOLD),

            "rf_version": versions["RandomForest"],
            "ridge_version": versions["Ridge"],
            "mlp_version": versions["NeuralNet"],
        })

    out_df = pd.DataFrame(rows)

    out_df_for_csv = out_df.copy()
    out_df_for_csv["timestamp_utc"] = out_df_for_csv["timestamp_utc"].astype(str)
    out_df_for_csv.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved locally: {OUTPUT_CSV}")

    save_predictions_to_feature_store(fs, out_df)

    return out_df, best_model, best_rmse


if __name__ == "__main__":
    run_inference_3days()
