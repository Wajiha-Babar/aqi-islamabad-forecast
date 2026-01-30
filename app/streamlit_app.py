import sys
from pathlib import Path
import os

import pandas as pd
import streamlit as st
import hopsworks
from dotenv import load_dotenv

load_dotenv()

# Add project root so we can import src.*
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.inference_3days import (
    run_inference_3days,
    OUTPUT_CSV,
    HAZARDOUS_THRESHOLD,
    PRED_FG_NAME,
    PRED_FG_VERSION,
)

HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST")

st.set_page_config(page_title="AQI Forecast (3 Models)", layout="wide")
st.title("🌫️ AQI Forecast — Next 3 Days (72 hours)")
st.caption("3 models predictions + best model (lowest RMSE from Hopsworks Model Registry) + hazardous alerts")


def load_predictions_from_hopsworks() -> pd.DataFrame:
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY,
        host=HOPSWORKS_HOST,
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(PRED_FG_NAME, version=PRED_FG_VERSION)
    df = fg.read()

    # Parse timestamps
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df = df.sort_values("timestamp_utc")

    return df


def load_predictions_fallback_csv() -> pd.DataFrame:
    df = pd.read_csv(OUTPUT_CSV)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df = df.sort_values("timestamp_utc")
    return df


# -----------------------------
# UI Controls
# -----------------------------
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Hazardous Threshold", f"{HAZARDOUS_THRESHOLD}")
with c2:
    st.write("")
with c3:
    run_now = st.button("▶ Run inference now (store to Hopsworks)")

if run_now:
    with st.spinner("Running inference..."):
        df_new, best_name, best_rmse = run_inference_3days()
    st.success(f"Done ✅ Best model: {best_name} | RMSE={best_rmse:.6f}")

st.divider()

# -----------------------------
# Load predictions
# -----------------------------
df = None
load_source = None

try:
    df = load_predictions_from_hopsworks()
    load_source = "Hopsworks Feature Store"
except Exception as e:
    st.warning(f"Could not load from Hopsworks, using CSV.\n\nError: {e}")
    if Path(OUTPUT_CSV).exists():
        df = load_predictions_fallback_csv()
        load_source = "Local CSV"
    else:
        df = None

if df is None or df.empty:
    st.error("No predictions found. Click **Run inference now**.")
    st.stop()

st.info(f"Loaded predictions from: **{load_source}** | Rows: {len(df)}")

# IMPORTANT: Hopsworks converted prediction column names to lowercase
# So we support both (CSV has uppercase, Hopsworks has lowercase)
def get_col(df, upper, lower):
    return lower if lower in df.columns else upper

col_rf = get_col(df, "pred_RandomForest", "pred_randomforest")
col_ridge = get_col(df, "pred_Ridge", "pred_ridge")
col_mlp = get_col(df, "pred_NeuralNet", "pred_neuralnet")

best_model = str(df["best_model"].iloc[-1])
best_rmse = float(df["best_model_rmse"].iloc[-1])

a, b = st.columns(2)
with a:
    st.subheader("🏆 Best Model")
    st.write(f"**Best model:** `{best_model}`")
    st.write(f"**RMSE:** `{best_rmse:.6f}`")
with b:
    st.subheader("📄 Download Output")
    st.download_button(
        "⬇ Download predictions.csv",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )

st.subheader("📈 Predictions (All 3 Models + Best)")
chart_df = df[["event_time", col_rf, col_ridge, col_mlp, "best_pred"]].set_index("event_time")
chart_df = chart_df.rename(columns={
    col_rf: "RandomForest",
    col_ridge: "Ridge",
    col_mlp: "NeuralNet",
    "best_pred": "Best"
})
st.line_chart(chart_df)

st.subheader("⚠️ Hazardous Alerts")
alerts = df[df["hazardous_alert"] == True][["event_time", "best_model", "best_pred"]]
if len(alerts) == 0:
    st.success(f"No hazardous alerts ✅ (threshold={HAZARDOUS_THRESHOLD})")
else:
    st.error(f"Hazardous alerts found: {len(alerts)} (AQI ≥ {HAZARDOUS_THRESHOLD})")
    st.dataframe(alerts, use_container_width=True)

st.subheader("🔎 Full Table")
st.dataframe(df, use_container_width=True)
