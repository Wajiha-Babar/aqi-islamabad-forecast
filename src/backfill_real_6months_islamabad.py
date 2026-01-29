from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd
import hopsworks
from dotenv import load_dotenv
load_dotenv()

from config import (
    CITY,
    OPENAQ_API_KEY,
    HOPSWORKS_PROJECT, HOPSWORKS_API_KEY, HOPSWORKS_HOST,
)
from features import build_feature_row
from feature_pipeline import to_dataframe, get_or_create_feature_group

LAT, LON = 33.6844, 73.0479  # Islamabad

# -------------------------
# AQI from PM2.5 (US EPA)
# -------------------------
def pm25_to_aqi_us_epa(pm25_ug_m3: float) -> Optional[float]:
    if pm25_ug_m3 is None:
        return None
    c = float(pm25_ug_m3)

    bps = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_lo, c_hi, i_lo, i_hi in bps:
        if c_lo <= c <= c_hi:
            aqi = (i_hi - i_lo) / (c_hi - c_lo) * (c - c_lo) + i_lo
            return float(round(aqi, 2))
    if c > 500.4:
        return 500.0
    return None


# -------------------------
# OpenAQ helpers
# -------------------------
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}

def openaq_get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not OPENAQ_API_KEY:
        raise ValueError("Missing OPENAQ_API_KEY. Put it in .env and config.py.")

    for attempt in range(1, 5):
        try:
            r = requests.get(
                url,
                params=params,
                headers={"X-API-Key": OPENAQ_API_KEY},
                timeout=60,
            )
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status in RETRY_STATUSES and attempt < 4:
                time.sleep(2 * attempt)
                continue
            raise
        except requests.exceptions.RequestException:
            if attempt < 4:
                time.sleep(2 * attempt)
                continue
            raise

    raise RuntimeError("OpenAQ request failed after retries.")


def list_candidate_locations(lat: float, lon: float, radius_m: int = 25000, limit: int = 100) -> List[Dict[str, Any]]:
    j = openaq_get(
        "https://api.openaq.org/v3/locations",
        {
            "coordinates": f"{lat:.4f},{lon:.4f}",
            "radius": min(int(radius_m), 25000),
            "parameters_id": [2],   # PM2.5
            "iso": "PK",
            "limit": limit,
            "page": 1,
        },
    )
    return j.get("results") or []


def sensor_id_from_location(location_obj: Dict[str, Any]) -> Optional[int]:
    sensors = location_obj.get("sensors") or []
    for s in sensors:
        if (s.get("parameter") or {}).get("name") == "pm25" and s.get("id") is not None:
            return int(s["id"])

    loc_id = location_obj.get("id")
    if loc_id is None:
        return None

    details = openaq_get(f"https://api.openaq.org/v3/locations/{loc_id}", {})
    loc_results = details.get("results") or []
    if not loc_results:
        return None

    for s in (loc_results[0].get("sensors") or []):
        if (s.get("parameter") or {}).get("name") == "pm25" and s.get("id") is not None:
            return int(s["id"])

    return None


def _fetch_hours_pages(sensor_id: int, dt_from: str, dt_to: str) -> List[Dict[str, Any]]:
    """
    ✅ MOST IMPORTANT FIX:
    /v3/sensors/{id}/hours uses:
      - datetime_from
      - datetime_to
    NOT date_from/date_to.
    """
    all_rows: List[Dict[str, Any]] = []
    page = 1
    limit = 200  # ✅ smaller = less timeout risk

    while True:
        j = openaq_get(
            f"https://api.openaq.org/v3/sensors/{sensor_id}/hours",
            {
                "datetime_from": dt_from,
                "datetime_to": dt_to,
                "limit": limit,
                "page": page,
            },
        )

        rows = j.get("results") or []
        if not rows:
            break

        for r in rows:
            period = r.get("period") or {}
            dt_to_val = ((period.get("datetimeTo") or {}).get("utc")) or ((r.get("datetime") or {}).get("utc"))
            val = r.get("value")
            if dt_to_val is None or val is None:
                continue
            all_rows.append({"timestamp_utc": dt_to_val, "pm25": val})

        if len(rows) < limit:
            break

        page += 1
        time.sleep(0.2)

    return all_rows


def fetch_pm25_hourly(sensor_id: int, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    ✅ 2-day chunks (more stable than 7 days in your case)
    """
    cursor = start_dt
    all_data: List[Dict[str, Any]] = []

    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=2), end_dt)

        dt_from = cursor.isoformat().replace("+00:00", "Z")
        dt_to = chunk_end.isoformat().replace("+00:00", "Z")

        print(f"  📥 Fetching PM2.5 chunk: {dt_from} -> {dt_to}")
        all_data.extend(_fetch_hours_pages(sensor_id, dt_from, dt_to))

        cursor = chunk_end
        time.sleep(0.2)

    df = pd.DataFrame(all_data)
    if df.empty:
        return pd.DataFrame(columns=["timestamp_utc", "pm25"])

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.floor("H")
    df = df.dropna(subset=["pm25"]).sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"])
    return df


def find_working_pm25_sensor(lat: float, lon: float, start_dt: datetime, end_dt: datetime) -> Tuple[int, str]:
    candidates = list_candidate_locations(lat, lon, radius_m=25000, limit=100)
    if not candidates:
        raise RuntimeError("No OpenAQ PM2.5 locations found within 25km of Islamabad.")

    for idx, loc in enumerate(candidates[:25]):
        loc_name = str(loc.get("name", f"location_{idx}"))
        sid = sensor_id_from_location(loc)
        if sid is None:
            continue

        print(f"Trying candidate sensor_id={sid} location='{loc_name}' ...")

        # ✅ very fast sensor test: only last 7 days
        test_start = max(start_dt, end_dt - timedelta(days=7))
        test_df = fetch_pm25_hourly(sid, test_start, end_dt)

        if len(test_df) > 10:
            print(f"✅ Selected sensor_id={sid} (test_rows={len(test_df)})")
            return sid, loc_name

        print(f"  -> Not enough data (test_rows={len(test_df)}), trying next...")

    raise RuntimeError("Tried multiple OpenAQ sensors near Islamabad but none had enough data.")


# -------------------------
# Open-Meteo weather (no key)
# -------------------------
def fetch_weather_open_meteo(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,windspeed_10m",
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()

    h = j.get("hourly") or {}
    times = h.get("time") or []
    if not times:
        return pd.DataFrame(columns=["timestamp_utc", "temp_c", "humidity", "pressure", "wind_speed"])

    # ✅ FIX: make it a pandas Series first (then .dt works)
    ts = pd.Series(pd.to_datetime(times, utc=True)).dt.floor("H")

    df = pd.DataFrame({
        "timestamp_utc": ts,
        "temp_c": h.get("temperature_2m"),
        "humidity": h.get("relative_humidity_2m"),
        "pressure": h.get("pressure_msl"),
        "wind_speed": h.get("windspeed_10m"),
    })

    df = df.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    return df


# -------------------------
# Main backfill
# -------------------------
def main(days: int = 183, insert_chunk: int = 500):
    end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days)

    start_date = start_dt.date().isoformat()
    end_date = end_dt.date().isoformat()

    print(f"REAL 6-month backfill city={CITY}")
    print(f"Range: {start_dt.isoformat()} -> {end_dt.isoformat()}")
    print("Step: hourly (OpenAQ hours + Open-Meteo hourly)")

    sensor_id, loc_name = find_working_pm25_sensor(LAT, LON, start_dt, end_dt)
    print(f"Using OpenAQ PM2.5 sensor_id={sensor_id} location='{loc_name}'")

    pm25_df = fetch_pm25_hourly(sensor_id, start_dt, end_dt)
    print("PM2.5 rows:", len(pm25_df))

    weather_df = fetch_weather_open_meteo(LAT, LON, start_date, end_date)
    print("Weather rows:", len(weather_df))

    merged = pd.merge(weather_df, pm25_df, on="timestamp_utc", how="inner")
    merged = merged.dropna(subset=["pm25"]).copy()
    merged["aqi"] = merged["pm25"].apply(pm25_to_aqi_us_epa)

    print("Merged usable rows:", len(merged))
    if merged.empty:
        raise RuntimeError(
            "No merged rows. Possible reasons:\n"
            "1) OpenAQ sensor has no data\n"
            "2) timestamps overlap issue\n"
        )

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY,
        host=HOPSWORKS_HOST,
    )
    fs = project.get_feature_store()
    fg = get_or_create_feature_group(fs)

    batch: List[pd.DataFrame] = []
    inserted = 0

    for _, r in merged.iterrows():
        aqicn_like = {"aqi": r["aqi"], "time": {"s": r["timestamp_utc"].strftime("%Y-%m-%d %H:%M:%S")}}
        openweather_like = {
            "main": {"temp": r["temp_c"], "humidity": r["humidity"], "pressure": r["pressure"]},
            "wind": {"speed": r["wind_speed"]},
        }
        row = build_feature_row(CITY, aqicn_like, openweather_like)
        batch.append(to_dataframe(row))

        if len(batch) >= insert_chunk:
            df_chunk = pd.concat(batch, ignore_index=True)
            fg.insert(df_chunk, write_options={"wait_for_job": True})
            inserted += len(df_chunk)
            print(f"✅ Inserted chunk: {len(df_chunk)} (total_inserted={inserted})")
            batch = []
            time.sleep(0.2)

    if batch:
        df_chunk = pd.concat(batch, ignore_index=True)
        fg.insert(df_chunk, write_options={"wait_for_job": True})
        inserted += len(df_chunk)
        print(f"✅ Inserted final chunk: {len(df_chunk)} (total_inserted={inserted})")

    print("✅ DONE: REAL 6-month backfill inserted into Hopsworks!")


if __name__ == "__main__":
    main(days=183, insert_chunk=500)
