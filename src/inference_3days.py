from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import joblib
import hopsworks


# -----------------------------
# SETTINGS (same project names)
# -----------------------------
FEATURE_GROUP_NAME = "aqi_features_v2"
FEATURE_GROUP_VERSION = 1

PRED_FG_NAME = "aqi_predictions_3days"
PRED_FG_VERSION = 1

CITY = "Islamabad"
LAT, LON = 33.6844, 73.0479

HAZARDOUS_THRESHOLD = 200

MODEL_NAMES = {
    "RandomForest": "aqi_random_forest",
    "Ridge": "aqi_ridge",
    "NeuralNet": "aqi_neural_net",
}

OUTPUT_DIR = Path("outputs")
OUTPUT_CSV = OUTPUT_DIR / "predictions.csv"


# -----------------------------
# Read env (your .env)
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
# Helpers
# -----------------------------
def ensure_outputs_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def next_72_hours_utc() -> pd.DatetimeIndex:
    """Next 72 hourly timestamps in UTC starting next full hour."""
    now = datetime.now(timezone.utc)
    start = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    # ✅ FIX: use "h" (not "H")
    return pd.date_range(start=start, periods=72, freq="h", tz="UTC")


def fetch_open_meteo_forecast_utc(lat: float, lon: float) -> pd.DataFrame:
    """
    Open-Meteo forecast (no API key). Hourly in UTC.
    """
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

    # ✅ FIX: use "h" (not "H")
    # ts = pd.to_datetime(times, utc=True).dt.floor("h")
    ts = pd.to_datetime(times, utc=True).floor("h")


    df = pd.DataFrame({
        "timestamp_utc": ts,
        "temp_c": h.get("temperature_2m"),
        "humidity": h.get("relative_humidity_2m"),
        "pressure": h.get("surface_pressure"),
        "wind_speed": h.get("windspeed_10m"),
    })

    df = df.dropna().drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    return df


def extract_rmse(model_meta) -> float:
    """Read RMSE from model registry metadata (training_metrics or metrics)."""
    for attr in ["training_metrics", "metrics"]:
        try:
            m = getattr(model_meta, attr, None)
            if isinstance(m, dict) and m:
                if "rmse" in m:
                    return float(m["rmse"])
                for k, v in m.items():
                    if "rmse" in str(k).lower():
                        return float(v)
        except Exception:
            pass
    return 1e18


def download_and_load_latest_model(mr, model_name: str):
    """
    Download latest version from Hopsworks Model Registry and load .pkl/.joblib.
    """
    versions = mr.get_models(model_name)
    if not versions:
        raise RuntimeError(f"No versions found for model '{model_name}'")

    latest = sorted(versions, key=lambda x: x.version)[-1]
    meta = mr.get_model(model_name, version=latest.version)
    model_dir = Path(meta.download())

    candidates = list(model_dir.rglob("*.pkl")) + list(model_dir.rglob("*.joblib"))
    if not candidates:
        raise RuntimeError(f"Downloaded model but no .pkl/.joblib found in: {model_dir}")

    model_obj = joblib.load(candidates[0])
    rmse = extract_rmse(meta)
    return model_obj, float(rmse), int(latest.version)


def load_history_from_feature_store(fs, days: int = 10) -> pd.DataFrame:
    """
    ✅ Hopsworks v4 FIX:
    Streamlit sometimes triggers Hive read -> not supported.
    So:
      - try fg.read()
      - if Hive error -> use fg.select_all().read() (Feature Query Service)
    """
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    # ✅ Try normal read
    try:
        df = fg.read()
    except ValueError as e:
        msg = str(e).lower()
        if "hive" in msg and "not supported" in msg:
            # ✅ Force FQS read (non-hive)
            df = fg.select_all().read()
        else:
            raise

    # ✅ FIX: use "h" not "H"
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.floor("h")
    df = df.sort_values("timestamp_utc").dropna(subset=["aqi"]).copy()

    end = df["timestamp_utc"].max()
    start = end - pd.Timedelta(days=days)
    df = df[df["timestamp_utc"] >= start].copy()

    if df.empty:
        raise RuntimeError("History is empty. Cannot build engineered features.")

    df = df[["timestamp_utc", "aqi"]].drop_duplicates(subset=["timestamp_utc"]).set_index("timestamp_utc").sort_index()

    # Fill hourly gaps
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    df = df.reindex(full_idx)
    df["aqi"] = df["aqi"].ffill().bfill()

    df = df.reset_index().rename(columns={"index": "timestamp_utc"})
    return df


def compute_engineered_features(history_aqi: pd.Series) -> Dict[str, float]:
    """
    Compute engineered features using history series (latest at end).
    """
    aqi_lag1 = float(history_aqi.iloc[-1])
    aqi_lag24 = float(history_aqi.iloc[-24]) if len(history_aqi) >= 24 else float(history_aqi.iloc[0])

    last24 = history_aqi.iloc[-24:] if len(history_aqi) >= 24 else history_aqi
    last7d = history_aqi.iloc[-168:] if len(history_aqi) >= 168 else history_aqi

    aqi_roll24 = float(last24.mean())
    aqi_roll7d = float(last7d.mean())
    aqi_diff1 = float(aqi_lag1 - float(history_aqi.iloc[-2])) if len(history_aqi) >= 2 else 0.0

    return {
        "aqi_lag1": aqi_lag1,
        "aqi_lag24": aqi_lag24,
        "aqi_roll24": aqi_roll24,
        "aqi_roll7d": aqi_roll7d,
        "aqi_diff1": aqi_diff1,
    }


def save_predictions_to_feature_store(fs, df_preds: pd.DataFrame):
    """
    Create/Use prediction Feature Group and insert 72 rows.
    PK must match your style: city + event_time (string).
    """
    fg = fs.get_or_create_feature_group(
        name=PRED_FG_NAME,
        version=PRED_FG_VERSION,
        primary_key=["city", "event_time"],
        description="Next 3 days AQI predictions (3 models + best model + alerts)",
        online_enabled=True,
    )

    if "event_time" not in df_preds.columns:
        df_preds["event_time"] = df_preds["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    print("✅ Inserting predictions into Hopsworks Feature Store...")
    fg.insert(df_preds, write_options={"wait_for_job": True})
    print("✅ Predictions stored in Feature Store!")


# -----------------------------
# MAIN INFERENCE
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

    print("✅ Step 2: Load last ~10 days AQI history (seed)...")
    hist_df = load_history_from_feature_store(fs, days=10)
    history_aqi = hist_df["aqi"].astype(float).copy()

    if len(history_aqi) < 50:
        raise RuntimeError("Not enough history rows for lag/rolling features.")

    print(f"   History rows used: {len(history_aqi)}")

    print("✅ Step 3: Load ALL 3 models from Model Registry (NO retraining)...")
    models = {}
    rmses = {}
    for pretty, reg in MODEL_NAMES.items():
        print(f"   -> Loading {pretty} ({reg})")
        model_obj, rmse, ver = download_and_load_latest_model(mr, reg)
        models[pretty] = model_obj
        rmses[pretty] = float(rmse)
        print(f"      version={ver} rmse={rmse}")

    print("✅ Step 4: Choose BEST model (lowest RMSE from registry metrics)...")
    best_model = sorted(rmses.items(), key=lambda kv: kv[1])[0][0]
    best_rmse = float(rmses[best_model])
    print(f"🏆 BEST: {best_model} | RMSE={best_rmse:.6f}")

    print("✅ Step 5: Fetch Open-Meteo forecast for next 72 hours...")
    weather_df = fetch_open_meteo_forecast_utc(LAT, LON)

    future_times = next_72_hours_utc()

    # align weather to future_times
    weather_72 = weather_df.set_index("timestamp_utc").reindex(future_times).ffill().bfill().reset_index()
    weather_72 = weather_72.rename(columns={"index": "timestamp_utc"})

    print("✅ Step 6: Recursive prediction for 72 hours (engineered features included)...")
    rows: List[Dict] = []

    for ts in future_times:
        w = weather_72[weather_72["timestamp_utc"] == ts].iloc[0]

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

        X_row = pd.DataFrame([{
            **base_features,
            **eng,
        }])

        preds = {}
        for pretty, model_obj in models.items():
            preds[pretty] = float(model_obj.predict(X_row)[0])

        best_pred = preds[best_model]

        history_aqi = pd.concat([history_aqi, pd.Series([best_pred])], ignore_index=True)

        event_time = ts.strftime("%Y-%m-%dT%H:%M:%SZ")

        rows.append({
            "timestamp_utc": ts,
            "event_time": event_time,
            "city": CITY,

            **base_features,
            **eng,

            "pred_RandomForest": preds["RandomForest"],
            "pred_Ridge": preds["Ridge"],
            "pred_NeuralNet": preds["NeuralNet"],

            "best_model": best_model,
            "best_model_rmse": best_rmse,
            "best_pred": best_pred,
            "hazardous_alert": bool(best_pred >= HAZARDOUS_THRESHOLD),
        })

    out_df = pd.DataFrame(rows)

    # Save local CSV (backup)
    out_df_for_csv = out_df.copy()
    out_df_for_csv["timestamp_utc"] = out_df_for_csv["timestamp_utc"].astype(str)
    out_df_for_csv.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved locally: {OUTPUT_CSV}")

    save_predictions_to_feature_store(fs, out_df)

    alerts = out_df[out_df["hazardous_alert"] == True]
    if len(alerts) > 0:
        print("\n⚠️ HAZARDOUS ALERTS (AQI >= threshold):")
        for _, r in alerts.iterrows():
            print(f" - {r['event_time']} AQI≈{r['best_pred']:.1f}")
    else:
        print("\n✅ No hazardous alerts in next 72 hours.")

    return out_df, best_model, best_rmse


if __name__ == "__main__":
    run_inference_3days()
