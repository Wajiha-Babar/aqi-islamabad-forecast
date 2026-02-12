🌫️ AQI Forecasting System — Islamabad (Next 72 Hours)

An end-to-end AQI forecasting system built with Hopsworks Feature Store + Model Registry.
It ingests hourly environmental features, trains multiple ML models, automatically selects the best model using RMSE, and generates hourly AQI forecasts for the next 72 hours. A deployed Streamlit dashboard visualizes trends, model performance, and hazardous air-quality alerts.

🚀 Live Deployed App

Streamlit Dashboard:
https://aqi-islamabad-forecast.streamlit.app/

✨ Key Features

✅ Hourly feature ingestion into Hopsworks Feature Store

✅ Daily training of 3 ML models:

Random Forest Regressor

Ridge Regression

Neural Network (MLP)

✅ Automatic best-model selection (lowest RMSE)

✅ 72-hour AQI forecasting (hourly predictions)

✅ Hazardous AQI alerts when AQI ≥ 200

✅ Predictions written back to Hopsworks Feature Group (online-enabled)

✅ Professional Streamlit dashboard with interactive charts + CSV export

🛠️ Tech Stack

Python

Hopsworks (Feature Store & Model Registry)

Scikit-learn

Open-Meteo Weather Forecast API

Streamlit

GitHub Actions (CI/CD)

🔄 Project Workflow
Step 1 — Feature Pipeline (Hourly Ingestion)

Fetches weather + AQI data, builds engineered features, and stores them in Hopsworks:

python src/feature_pipeline.py

Step 2 — Model Training (Daily)

Trains all models and uploads them to the Hopsworks Model Registry:

python src/train_models.py

Step 3 — Inference Pipeline (Next 72 Hours Forecast)

Loads latest model versions, selects the best model, and generates predictions:

python src/inference_3days.py


Outputs:

outputs/predictions_3days.csv

Hopsworks Feature Group: aqi_predictions_3days (version controlled via PRED_FG_VERSION)

Step 4 — Streamlit Dashboard

Run locally:

streamlit run app/streamlit_app.py


Or open deployed app:
https://aqi-islamabad-forecast.streamlit.app/

⚙️ Setup

Requirements: Python 3.10+
Install dependencies:

pip install -r requirements.txt


Make sure Hopsworks credentials are configured via .env (local) or Streamlit Secrets (deployment).

📊 Dashboard Highlights

Model performance comparison (RMSE)

Best model selection + reason

AQI trend (observed vs forecast)

Hazardous air-quality alerts

Forecast export (CSV)

👩‍💻 Author

Wajiha Babar
Department of Software Engineering