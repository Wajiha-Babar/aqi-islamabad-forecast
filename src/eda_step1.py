import hopsworks
import pandas as pd
from config import FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION

print("1) Login Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

print("2) Read Feature Group...")
fg = fs.get_feature_group(FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION)
df = fg.read()

print("\n✅ (A) Data shape (rows, cols):", df.shape)
print("\n✅ (B) Columns:", list(df.columns))
print("\n✅ (C) First 5 rows:")
print(df.head())

# timestamp ko datetime banado
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

print("\n✅ Timestamp min:", df["timestamp_utc"].min())
print("✅ Timestamp max:", df["timestamp_utc"].max())
