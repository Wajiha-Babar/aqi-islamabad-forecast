# 🌫️ AQI Forecasting System — Islamabad (Next 72 Hours)
This project is an **end-to-end Air Quality Index (AQI) forecasting system** built using the **Hopsworks Feature Store** and **Model Registry**.
It automatically collects real-time environmental data, trains multiple machine learning models, selects the best-performing model based on RMSE, and generates hourly AQI predictions for the next **72 hours**.
A fully deployed **Streamlit dashboard** provides interactive visualizations, model comparison, and hazardous air-quality alerts.
---
## 🚀 Live Deployed App

🔗 Streamlit Dashboard:  
https://aqi-islamabad-forecast.streamlit.app/

---
## 📌 Key Features
✅ Hourly AQI feature ingestion using Hopsworks Feature Store  
✅ Daily training of 3 ML models:
- Random Forest Regressor  
- Ridge Regression  
- Neural Network (MLP)

✅ Automatic **Best Model Selection** (lowest RMSE)  
✅ Next **72-hour AQI Forecasting**  
✅ Hazardous AQI alerts when AQI ≥ 200  
✅ Predictions stored back into Hopsworks Feature Group  
✅ Professional Streamlit Dashboard for visualization & download

---

## 🛠️ Tech Stack

- **Python**
- **Hopsworks Feature Store & Model Registry**
- **Scikit-learn**
- **Open-Meteo Weather Forecast API**
- **Streamlit Dashboard**
- **GitHub Actions (CI/CD Automation)**

---

## 📂 Project Workflow

### Step 1 — Feature Pipeline (Hourly Data Ingestion)

Fetches weather + AQI data and stores engineered features into Hopsworks:

```bash
python src/feature_pipeline.py
Step 2 — Model Training (Daily)
Trains all three models and uploads them into the Hopsworks Model Registry:
python src/train_models.py
Step 3 — Inference Pipeline (Next 72 Hours Forecast)
Loads the latest models from registry, selects the best model, and generates predictions:
python src/inference_3days.py
Predictions are saved to:

outputs/predictions_3days.csv

Hopsworks Feature Group: aqi_predictions_3days

Step 4 — Launch Streamlit Dashboard

Run locally:

streamlit run app/streamlit_app.py

Or view deployed version here:

https://aqi-islamabad-forecast.streamlit.app/

⚙️ Installation Requirements
Make sure Python 3.10+ is installed.
Install dependencies:
pip install -r requirements.txt
📊 Output Example

## 📊 Output Example
## 📊 Output Example

![Dashboard](https://raw.githubusercontent.com/Wajiha-Babar/aqi-islamabad-forecast/main/assets/dashboard.png)

![Dashboard 1](https://raw.githubusercontent.com/Wajiha-Babar/aqi-islamabad-forecast/main/assets/dashboard_1.jpeg)

![Dashboard 2](https://raw.githubusercontent.com/Wajiha-Babar/aqi-islamabad-forecast/main/assets/dashboard_2.jpeg)


The dashboard provides:

- Model performance comparison  
- Best model RMSE  
- AQI prediction trends  
- Hazardous air quality alerts  
- CSV export option  
👩‍💻 Author
Wajiha Babar
Department of Software Engineering
