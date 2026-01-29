# src/features.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class FeatureRow:
    timestamp_utc: datetime
    city: str

    # time features
    hour: int
    day: int
    month: int
    day_of_week: int

    # weather features
    temp_c: Optional[float]
    humidity: Optional[float]
    pressure: Optional[float]
    wind_speed: Optional[float]

    # target
    aqi: Optional[float]


def _parse_aqicn_time(aqicn_data: Dict[str, Any]) -> datetime:
    """
    AQICN time sometimes comes like: aqicn_data["time"]["s"] = "2025-02-18 18:00:00"
    We'll treat it as UTC if timezone info missing.
    """
    t = (aqicn_data.get("time") or {}).get("s")
    if not t:
        return datetime.now(timezone.utc)

    # format: "YYYY-MM-DD HH:MM:SS"
    dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
    return dt.replace(tzinfo=timezone.utc)


def build_feature_row(
    city: str,
    aqicn_data: Dict[str, Any],
    openweather_data: Dict[str, Any],
) -> FeatureRow:
    ts = _parse_aqicn_time(aqicn_data)

    hour = ts.hour
    day = ts.day
    month = ts.month
    day_of_week = ts.weekday()  # Mon=0 ... Sun=6

    main = openweather_data.get("main") or {}
    wind = openweather_data.get("wind") or {}

    temp_c = main.get("temp")
    humidity = main.get("humidity")
    pressure = main.get("pressure")
    wind_speed = wind.get("speed")

    aqi = aqicn_data.get("aqi")

    # Sometimes AQI can be "-" or None
    try:
        aqi = float(aqi) if aqi is not None and aqi != "-" else None
    except Exception:
        aqi = None

    return FeatureRow(
        timestamp_utc=ts,
        city=city,
        hour=hour,
        day=day,
        month=month,
        day_of_week=day_of_week,
        temp_c=float(temp_c) if temp_c is not None else None,
        humidity=float(humidity) if humidity is not None else None,
        pressure=float(pressure) if pressure is not None else None,
        wind_speed=float(wind_speed) if wind_speed is not None else None,
        aqi=aqi,
    )


if __name__ == "__main__":
    # quick manual test (ye sirf test ke liye hai)
    sample_aqicn = {
        "aqi": 34,
        "time": {"s": "2025-02-18 18:00:00"},
    }
    sample_weather = {
        "main": {"temp": 14.99, "humidity": 58, "pressure": 1016},
        "wind": {"speed": 2.1},
    }

    row = build_feature_row("Islamabad", sample_aqicn, sample_weather)
    print(row)
