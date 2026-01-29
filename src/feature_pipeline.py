# src/feature_pipeline.py
from dotenv import load_dotenv
load_dotenv()

import hopsworks
import pandas as pd

from fetch_data import fetch_aqicn_aqi, fetch_openweather_current
from features import build_feature_row
from config import (
    CITY,
    HOPSWORKS_PROJECT,
    HOPSWORKS_API_KEY,
    HOPSWORKS_HOST,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)


def _require_env():
    missing = []
    if not HOPSWORKS_PROJECT:
        missing.append("HOPSWORKS_PROJECT")
    if not HOPSWORKS_API_KEY:
        missing.append("HOPSWORKS_API_KEY")
    if not HOPSWORKS_HOST:
        missing.append("HOPSWORKS_HOST")
    if missing:
        raise ValueError(f"Missing env vars: {', '.join(missing)}")


def to_dataframe(feature_row):
    """
    Online feature store primary keys do not support timestamp type as PK (in your case),
    so we create a string key 'event_time' and use that as primary key.
    We still keep timestamp_utc for offline training.
    """
    event_time = feature_row.timestamp_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    return pd.DataFrame([{
        # Offline/training time column
        "timestamp_utc": feature_row.timestamp_utc,

        # ✅ String PK for online store
        "event_time": event_time,

        "city": feature_row.city,
        "hour": feature_row.hour,
        "day": feature_row.day,
        "month": feature_row.month,
        "day_of_week": feature_row.day_of_week,
        "temp_c": feature_row.temp_c,
        "humidity": feature_row.humidity,
        "pressure": feature_row.pressure,
        "wind_speed": feature_row.wind_speed,
        "aqi": feature_row.aqi,
    }])


def get_or_create_feature_group(fs):
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,

        # ✅ FIX: use string PK instead of timestamp_utc
        primary_key=["city", "event_time"],

        description="AQI + weather features (hourly)",
        online_enabled=True,
    )
    return fg


def run(lat: float, lon: float):
    _require_env()

    print("Fetching raw data...")
    aqicn = fetch_aqicn_aqi(CITY)
    weather = fetch_openweather_current(lat, lon)

    print("Building features...")
    row = build_feature_row(CITY, aqicn, weather)
    df = to_dataframe(row)

    print("✅ DataFrame to insert:")
    print(df)

    print("Logging in to Hopsworks...")
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY,
        host=HOPSWORKS_HOST,
    )
    fs = project.get_feature_store()

    print("Getting/creating feature group...")
    fg = get_or_create_feature_group(fs)

    print("Inserting into feature store...")
    fg.insert(df, write_options={"wait_for_job": True})

    print("✅ Insert done!")


if __name__ == "__main__":
    # Islamabad coordinates
    LAT, LON = 33.6844, 73.0479
    run(LAT, LON)
