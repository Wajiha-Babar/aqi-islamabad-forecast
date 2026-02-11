import sys
from pathlib import Path
import os

import pandas as pd
import streamlit as st
import hopsworks
from dotenv import load_dotenv

# ✅ Safe: locally .env load, cloud me ignore (no crash)
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

# -----------------------------
# ENV (✅ Works on Streamlit Cloud + Local)
# - Streamlit Cloud: st.secrets
# - Local: .env via os.getenv
# -----------------------------
def _get_secret(key: str, default=None):
    # Streamlit Cloud secrets first
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    # fallback to env
    return os.getenv(key, default)

HOPSWORKS_PROJECT = _get_secret("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = _get_secret("HOPSWORKS_API_KEY")
HOPSWORKS_HOST = _get_secret("HOPSWORKS_HOST")

if not HOPSWORKS_PROJECT or not HOPSWORKS_API_KEY or not HOPSWORKS_HOST:
    st.error("❌ Missing Hopsworks secrets/env vars.")
    st.stop()

# -----------------------------
# PAGE (Premium)
# -----------------------------
st.set_page_config(
    page_title="AQI Forecast — Islamabad",
    page_icon="🌫️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      div[data-testid="stMetricValue"] { font-size: 2.0rem; }
      div[data-testid="stMetricLabel"] { opacity: .85; }
      .soft-card {
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 16px;
        padding: 14px 16px;
        background: rgba(255,255,255,.03);
      }
      .subtle { opacity:.85; }
      .tiny { font-size: .92rem; opacity: .78; }
      .pill {
        display:inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,.12);
        background: rgba(255,255,255,.05);
        font-size: 0.9rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌫️ AQI Forecast — Next 72 Hours")
st.caption(
    "Three-model forecasting + automatic best-model selection (lowest RMSE from Model Registry) + hazardous alerts."
)

# -----------------------------
# Hopsworks login (CACHED!)
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_hopsworks_project():
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

    if "timestamp_utc" not in df.columns:
        raise RuntimeError("Loaded data missing 'timestamp_utc'")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

    if "event_time" not in df.columns:
        df["event_time"] = df["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")

    if "hazardous_alert" in df.columns:
        if df["hazardous_alert"].dtype != bool:
            df["hazardous_alert"] = (
                df["hazardous_alert"].astype(str).str.lower().isin(["true", "1", "yes"])
            )
    else:
        if "best_pred" in df.columns:
            df["hazardous_alert"] = df["best_pred"].astype(float) >= float(HAZARDOUS_THRESHOLD)

    df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)
    return df


def load_predictions_from_hopsworks() -> pd.DataFrame:
    project = get_hopsworks_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(PRED_FG_NAME, version=PRED_FG_VERSION)

    try:
        df = fg.read(online=True)
    except TypeError:
        df = fg.read()
    except Exception:
        df = fg.select_all().read()

    return _normalize_df(df)


def load_predictions_fallback_csv() -> pd.DataFrame:
    if not Path(OUTPUT_CSV).exists():
        raise FileNotFoundError(f"CSV not found: {OUTPUT_CSV}")
    df = pd.read_csv(OUTPUT_CSV)
    return _normalize_df(df)

# -----------------------------
# Sidebar (Premium controls)
# -----------------------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    run_now = st.button("▶ Run inference now", use_container_width=True)
    st.markdown("---")
    st.markdown("### 🧭 Data Source")
    st.caption("We load from Hopsworks first; if not available, we fallback to local CSV.")
    show_table = st.toggle("Show full table", value=True)
    st.markdown("---")
    st.markdown("### ℹ️ Threshold")
    st.write(f"**Hazardous AQI threshold:** `{HAZARDOUS_THRESHOLD}`")

# -----------------------------
# Run inference action
# -----------------------------
if run_now:
    with st.spinner("Running inference + storing to Hopsworks..."):
        df_new, best_name, best_rmse = run_inference_3days()
    st.success(f"Done ✅ Best model: {best_name} | RMSE={best_rmse:.6f}")

# -----------------------------
# Load predictions (Hopsworks -> CSV fallback)
# -----------------------------
df = None
load_source = None

try:
    df = load_predictions_from_hopsworks()
    load_source = f"Hopsworks Feature Group: {PRED_FG_NAME} (v{PRED_FG_VERSION})"
except Exception as e:
    st.warning(f"Could not load from Hopsworks; using CSV fallback.\n\nError: {e}")
    df = load_predictions_fallback_csv()
    load_source = f"Local CSV ({OUTPUT_CSV})"

if df is None or df.empty:
    st.error("No predictions found. Click **Run inference now**.")
    st.stop()

# -----------------------------
# Column checks
# -----------------------------
col_rf = "pred_randomforest"
col_ridge = "pred_ridge"
col_mlp = "pred_neuralnet"

required = ["best_pred", "best_model", "best_model_rmse", col_rf, col_ridge, col_mlp]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns in loaded data: {missing}")
    st.stop()

# -----------------------------
# Premium header cards
# -----------------------------
best_model = str(df["best_model"].iloc[-1])
best_rmse = float(df["best_model_rmse"].iloc[-1])
latest_ts = df["timestamp_utc"].max()
haz_count = int((df["hazardous_alert"] == True).sum())

top = st.columns([1.2, 1.2, 1.2, 1.2])
top[0].metric("Hazardous Threshold", f"{HAZARDOUS_THRESHOLD}")
top[1].metric("Rows Loaded", f"{len(df)}")
top[2].metric("Best Model", best_model)
top[3].metric("Best RMSE", f"{best_rmse:.4f}")

st.markdown(
    f"""
    <div class="soft-card">
      <div class="subtle">Data source</div>
      <div><span class="pill">{load_source}</span></div>
      <div class="tiny">Last timestamp (UTC): <b>{latest_ts}</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# -----------------------------
# Download + quick summary
# -----------------------------
a, b, c = st.columns([1.4, 1.0, 1.0])
with a:
    st.subheader("📄 Export")
    st.download_button(
        "⬇ Download predictions.csv",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
with b:
    st.subheader("⚠️ Alerts")
    if haz_count == 0:
        st.success("No hazardous hours ✅")
    else:
        st.error(f"{haz_count} hazardous hours")
with c:
    st.subheader("🕒 Horizon")
    st.info("Next 72 hours", icon="🗓️")

st.divider()

# -----------------------------
# Charts
# -----------------------------
st.subheader("📈 Predictions — All Models vs Best")

chart_df = df[["event_time", col_rf, col_ridge, col_mlp, "best_pred"]].copy()
chart_df["event_time"] = pd.to_datetime(chart_df["event_time"], utc=True, errors="coerce")
chart_df = chart_df.dropna(subset=["event_time"]).set_index("event_time")

chart_df = chart_df.rename(columns={
    col_rf: "RandomForest",
    col_ridge: "Ridge",
    col_mlp: "NeuralNet",
    "best_pred": "Best",
})

st.line_chart(chart_df)

# -----------------------------
# Alerts table (Premium)
# -----------------------------
st.subheader("🚨 Hazardous Alerts (AQI ≥ Threshold)")
alerts = df[df["hazardous_alert"] == True][["event_time", "best_model", "best_pred"]].copy()

if alerts.empty:
    st.success(f"No hazardous alerts ✅ (threshold={HAZARDOUS_THRESHOLD})")
else:
    st.error(f"Hazardous alerts found: {len(alerts)}")
    st.dataframe(alerts, use_container_width=True, hide_index=True)

# -----------------------------
# Full table (toggle)
# -----------------------------
if show_table:
    st.subheader("🔎 Full Prediction Table")
    st.dataframe(df, use_container_width=True, hide_index=True)
