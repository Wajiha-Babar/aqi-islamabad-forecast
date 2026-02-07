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
    PRED_FG_VERSION,  # make sure this is 3 in inference_3days.py
)

HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST")

if not HOPSWORKS_PROJECT or not HOPSWORKS_API_KEY or not HOPSWORKS_HOST:
    raise ValueError("Missing Hopsworks env vars. Check .env")

st.set_page_config(page_title="AQI Forecast (3 Models)", layout="wide")
st.title("🌫️ AQI Forecast — Next 3 Days (72 hours)")
st.caption("3 models predictions + best model (lowest RMSE from Model Registry) + hazardous alerts")


# -----------------------------
# Hopsworks login (CACHED!)
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_hopsworks_project():
    # ✅ cache login to avoid streamlit rerun breaking client state
    return hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY,
        host=HOPSWORKS_HOST,
    )


# -----------------------------
# Loaders
# -----------------------------
def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # timestamp_utc must exist
    if "timestamp_utc" not in df.columns:
        raise RuntimeError("Loaded data missing 'timestamp_utc'")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

    # event_time optional -> create
    if "event_time" not in df.columns:
        df["event_time"] = df["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")

    # hazardous_alert normalize
    if "hazardous_alert" in df.columns:
        # supports True/False, 0/1, "True"/"False"
        if df["hazardous_alert"].dtype != bool:
            df["hazardous_alert"] = df["hazardous_alert"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        # derive if missing
        if "best_pred" in df.columns:
            df["hazardous_alert"] = df["best_pred"].astype(float) >= float(HAZARDOUS_THRESHOLD)

    df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    return df


def load_predictions_from_hopsworks() -> pd.DataFrame:
    project = get_hopsworks_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(PRED_FG_NAME, version=PRED_FG_VERSION)

    # ✅ IMPORTANT: try ONLINE read first (avoids Hudi/Hive offline errors)
    try:
        df = fg.read(online=True)
    except TypeError:
        # older hsfs may not support online=True
        df = fg.read()
    except Exception:
        # try Feature Query Service as fallback (offline)
        try:
            df = fg.select_all().read()
        except Exception as e:
            # re-raise to be caught by main try/except
            raise e

    return _normalize_df(df)


def load_predictions_fallback_csv() -> pd.DataFrame:
    if not Path(OUTPUT_CSV).exists():
        raise FileNotFoundError(f"CSV not found: {OUTPUT_CSV}")

    df = pd.read_csv(OUTPUT_CSV)
    return _normalize_df(df)


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
# Load predictions (Hopsworks -> CSV fallback)
# -----------------------------
df = None
load_source = None

try:
    df = load_predictions_from_hopsworks()
    load_source = f"Hopsworks (FG={PRED_FG_NAME} v{PRED_FG_VERSION})"
except Exception as e:
    st.warning(f"Could not load from Hopsworks, using CSV.\n\nError: {e}")
    df = load_predictions_fallback_csv()
    load_source = f"Local CSV ({OUTPUT_CSV})"

if df is None or df.empty:
    st.error("No predictions found. Click **Run inference now**.")
    st.stop()

st.info(f"Loaded predictions from: **{load_source}** | Rows: {len(df)}")


# -----------------------------
# Columns (lowercase)
# -----------------------------
# Your inference writes these (after our fixes)
col_rf = "pred_randomforest"
col_ridge = "pred_ridge"
col_mlp = "pred_neuralnet"

missing = [c for c in ["best_pred", "best_model", "best_model_rmse", col_rf, col_ridge, col_mlp] if c not in df.columns]
if missing:
    st.error(f"Missing columns in loaded data: {missing}")
    st.stop()

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
