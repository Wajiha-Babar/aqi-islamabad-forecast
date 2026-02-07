import hopsworks
import pandas as pd
from config import FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group(FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION)
df = fg.read()

df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
df = df.sort_values("timestamp_utc")

# ✅ 1) AQI lag 1 hour
df["aqi_lag1"] = df["aqi"].shift(1)

# ✅ 2) AQI lag 24 hour (previous day)
df["aqi_lag24"] = df["aqi"].shift(24)

# ✅ 3) Rolling 24h mean
df["aqi_roll24"] = df["aqi"].rolling(24).mean()

# ✅ 4) Rolling 7 day mean
df["aqi_roll7d"] = df["aqi"].rolling(24*7).mean()

# ✅ 5) AQI change rate
# df["aqi_diff1"] = df["aqi"] - df["aqi_lag1"]
df["aqi_lag2"] = df["aqi"].shift(2)
df["aqi_diff1"] = df["aqi_lag1"] - df["aqi_lag2"]

print(df[["aqi","aqi_lag1","aqi_lag24","aqi_roll24","aqi_roll7d","aqi_diff1"]].head(50))
