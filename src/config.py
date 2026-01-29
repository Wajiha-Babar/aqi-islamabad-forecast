# src/config.py
import os

CITY = os.getenv("CITY", "Islamabad")

# Hopsworks
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST")

FEATURE_GROUP_VERSION = int(os.getenv("FEATURE_GROUP_VERSION", "1"))
FEATURE_GROUP_NAME = os.getenv("FEATURE_GROUP_NAME", "aqi_features_v2")

# OpenWeather
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
# OpenAQ
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")

