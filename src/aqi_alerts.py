import pandas as pd

# Suppose df loaded
df = pd.read_csv("aqi_data.csv")  # or hopsworks

def check_alert(aqi):
    if aqi > 300:
        return "🚨 Severe Hazardous"
    elif aqi > 150:
        return "⚠️ Unhealthy"
    elif aqi > 100:
        return "😷 Moderate"
    else:
        return "😊 Good"

df["alert"] = df["aqi"].apply(check_alert)
print(df[["timestamp_utc","aqi","alert"]].tail(20))
