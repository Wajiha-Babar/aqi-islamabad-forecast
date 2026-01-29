import hopsworks
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from config import FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group(FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION)
df = fg.read()

df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
df = df.sort_values("timestamp_utc")

# Feature Engineering (same)
df["aqi_lag1"] = df["aqi"].shift(1)
df["aqi_lag24"] = df["aqi"].shift(24)
df["aqi_roll24"] = df["aqi"].rolling(24).mean()
df["aqi_diff1"] = df["aqi"] - df["aqi_lag1"]

df = df.dropna()

X = df[["temp_c","humidity","pressure","wind_speed","aqi_lag1","aqi_lag24","aqi_roll24","aqi_diff1"]]
y = df["aqi"]

model = RandomForestRegressor()
model.fit(X, y)

explainer = shap.Explainer(model, X)
shap_values = explainer(X)

shap.plots.bar(shap_values)
