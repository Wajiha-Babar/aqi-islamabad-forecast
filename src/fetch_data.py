# src/fetch_data.py
import os
import requests


def fetch_aqicn_aqi(city: str) -> dict:
    """
    Fetch current AQI data from AQICN by city name.
    Requires env var: AQICN_TOKEN
    """
    token = os.getenv("AQICN_TOKEN")
    if not token:
        raise ValueError("Missing AQICN_TOKEN in environment variables / GitHub Secrets")

    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    if data.get("status") != "ok":
        raise RuntimeError(f"AQICN API returned error: {data}")

    return data["data"]


def fetch_openweather_current(lat: float, lon: float, units: str = "metric") -> dict:
    """
    Fetch current weather from OpenWeather by lat/lon.
    Requires env var: OPENWEATHER_API_KEY
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENWEATHER_API_KEY in environment variables / GitHub Secrets")

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": units}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


# ✅ IMPORTANT: TEST CODE MUST BE INSIDE MAIN BLOCK
if __name__ == "__main__":
    # Islamabad config
    CITY = "Islamabad"
    LAT, LON = 33.6844, 73.0479

    print("Fetching AQI from AQICN...")
    aqicn = fetch_aqicn_aqi(CITY)
    print("✅ AQICN fetched. Sample:")
    print("AQI:", aqicn.get("aqi"))
    print("Time:", aqicn.get("time", {}).get("s"))

    print("\nFetching Weather from OpenWeather...")
    weather = fetch_openweather_current(LAT, LON)
    print("✅ OpenWeather fetched. Sample:")
    print("Temp:", weather.get("main", {}).get("temp"))
    print("Humidity:", weather.get("main", {}).get("humidity"))
    print("Weather:", (weather.get("weather") or [{}])[0].get("description"))

    # Feature engineering test
    from features import build_feature_row

    row = build_feature_row(CITY, aqicn, weather)
    print("\n✅ Feature Row created:")
    print(row)
