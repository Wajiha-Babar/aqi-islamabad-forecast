This project is an end-to-end AQI forecasting system built using Hopsworks Feature Store and Model Registry.  
It trains three machine learning models daily and automatically selects the best-performing model (lowest RMSE) during inference.  
The system generates hourly AQI predictions for the next 72 hours and highlights hazardous air quality alerts.  
A Streamlit dashboard is used to visualize predictions and model performance.

## Tech Stack
- Python
- Hopsworks Feature Store & Model Registry
- Scikit-learn
- Open-Meteo (weather forecast)
- Streamlit

## Requirements Installation

Make sure Python 3.10+ is installed, then run:

```bash
pip install -r requirements.txt
How to Run the Project
1. Run Feature Pipeline (hourly data ingestion)
python src/feature_pipeline.py
2. Train Models (daily training + registry upload)
python src/train_models.py
python src/train_models.py
python src/inference_3days.py
4. Launch Dashboard
streamlit run app/streamlit_app.py
