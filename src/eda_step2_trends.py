import hopsworks
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config import FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION

# ✅ output folder
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group(FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION)
df = fg.read()

df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
df = df.sort_values("timestamp_utc").set_index("timestamp_utc")

# 1) Daily AQI trend
daily = df["aqi"].resample("D").mean()

plt.figure()
daily.plot()
plt.title("Daily Average AQI Trend (Islamabad)")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.tight_layout()
plt.savefig(OUT / "daily_aqi_trend.png", dpi=200)
plt.close()

# 2) Monthly AQI trend
monthly = df["aqi"].resample("M").mean()

plt.figure()
monthly.plot()
plt.title("Monthly Average AQI Trend (Islamabad)")
plt.xlabel("Month")
plt.ylabel("AQI")
plt.tight_layout()
plt.savefig(OUT / "monthly_aqi_trend.png", dpi=200)
plt.close()

# 3) Temp vs AQI scatter
plt.figure()
plt.scatter(df["temp_c"], df["aqi"], alpha=0.25)
plt.title("Temperature vs AQI (Islamabad)")
plt.xlabel("Temperature (C)")
plt.ylabel("AQI")
plt.tight_layout()
plt.savefig(OUT / "temp_vs_aqi.png", dpi=200)
plt.close()

print("✅ Saved 3 graphs in outputs/ folder:")
print(" - outputs/daily_aqi_trend.png")
print(" - outputs/monthly_aqi_trend.png")
print(" - outputs/temp_vs_aqi.png")
