# app/streamlit_app.py
import sys
import time
import numpy as np
from pathlib import Path
import os

import pandas as pd
import streamlit as st
import hopsworks
from dotenv import load_dotenv

import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

load_dotenv()

# -----------------------------
# Project import path
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Only need constants (NO inference run inside Streamlit for speed)
from src.inference_3days import HAZARDOUS_THRESHOLD, PRED_FG_NAME

#  versions from env/secrets (keep consistent with GitHub Actions)
PRED_FG_VERSION = int(os.getenv("PRED_FG_VERSION", "1"))
FEATURE_GROUP_NAME = os.getenv("HOPSWORKS_FEATURE_GROUP", "aqi_features_v2")
FEATURE_GROUP_VERSION = int(os.getenv("HOPSWORKS_FEATURE_GROUP_VERSION", "1"))

# -----------------------------
# Secrets helper
# -----------------------------
def _get_secret(key: str, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

HOPSWORKS_PROJECT = _get_secret("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = _get_secret("HOPSWORKS_API_KEY")
HOPSWORKS_HOST = _get_secret("HOPSWORKS_HOST")

st.set_page_config(
    page_title="AQI Intelligence — Islamabad",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# UI state 
# -----------------------------
if "tab" not in st.session_state:
    st.session_state.tab = "🧾 Forecast Table"

# -----------------------------
# THEME (same)
# -----------------------------
st.markdown(
    """
<style>
.stApp{
  background:
    radial-gradient(circle at 12% 8%, rgba(56,189,248,0.18), transparent 40%),
    radial-gradient(circle at 88% 18%, rgba(99,102,241,0.14), transparent 42%),
    radial-gradient(circle at 55% 92%, rgba(14,165,233,0.10), transparent 48%),
    linear-gradient(180deg, #050913 0%, #071125 55%, #050913 100%);
}
.block-container{max-width:1250px; padding-top:1.05rem; padding-bottom:2.2rem;}

section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  border-right: 1px solid rgba(255,255,255,0.10);
}
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.92); }

.hero{
  border-radius:18px; padding:18px 18px;
  border:1px solid rgba(255,255,255,0.14);
  background: linear-gradient(90deg, rgba(56,189,248,0.68), rgba(99,102,241,0.35));
  box-shadow: 0 16px 52px rgba(0,0,0,0.55);
}
.hero h1{margin:0; font-size:2.05rem; color:#fff; letter-spacing:.2px;}
.hero p{margin:6px 0 0 0; opacity:.92; color:rgba(255,255,255,0.92);}

.card{
  border:1px solid rgba(255,255,255,0.12);
  border-radius:16px;
  padding:14px 16px;
  background: rgba(255,255,255,0.045);
  box-shadow: 0 10px 28px rgba(0,0,0,0.38);
}
.card-title{font-size:.92rem; opacity:.85; margin-bottom:6px;}
.card-value{font-size:1.55rem; font-weight:850; color:#ffffff;}
.card-sub{font-size:.92rem; opacity:.82; margin-top:6px; line-height:1.35;}

.pill{
  display:inline-block; padding:6px 10px; border-radius:999px;
  border:1px solid rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.07);
  font-weight:800;
}

.big-btn .stDownloadButton button,
.big-btn .stButton button{
  width:100% !important;
  border-radius:14px !important;
  border:1px solid rgba(255,255,255,0.18) !important;
  background: linear-gradient(90deg, rgba(56,189,248,0.88), rgba(99,102,241,0.58)) !important;
  color:white !important;
  font-weight:850 !important;
  padding:0.90rem 1rem !important;
}
.small-note{ font-size:.86rem; opacity:.78; }
.kpi{ color: rgba(56,189,248,0.95); font-weight:850; }
</style>
""",
    unsafe_allow_html=True,
)

if not HOPSWORKS_PROJECT or not HOPSWORKS_API_KEY or not HOPSWORKS_HOST:
    st.error("❌ Missing Hopsworks secrets/env vars. Add them in Streamlit Cloud → Secrets.")
    st.stop()

# -----------------------------
# Helpers
# -----------------------------
def fmt(x, d=0, suffix=""):
    if x is None or pd.isna(x):
        return "—"
    try:
        return f"{float(x):.{d}f}{suffix}"
    except Exception:
        return "—"

def aqi_category(aqi: float):
    if aqi is None or pd.isna(aqi):
        return ("N/A", "⚪", "N/A", "No AQI data available.")
    aqi = float(aqi)
    if aqi <= 50:
        return ("Good", "🟢", "Good", "Air quality is clean and comfortable for outdoor activities.")
    if aqi <= 100:
        return ("Moderate", "🟡", "Moderate", "Air quality is acceptable; sensitive individuals should take care.")
    if aqi <= 150:
        return ("Unhealthy (Sensitive)", "🟠", "Unhealthy for Sensitive Groups", "Sensitive groups should limit outdoor exertion; a mask may help.")
    if aqi <= 200:
        return ("Unhealthy", "🔴", "Unhealthy", "Reduce outdoor activity and prefer indoor environments.")
    if aqi <= 300:
        return ("Very Unhealthy", "🟣", "Very Unhealthy", "Stay indoors where possible; consider air filtration.")
    return ("Hazardous", "🟤", "Hazardous", "Health alert: avoid going outside and follow strict precautions.")

@st.cache_resource(show_spinner=False)
def _hw_project_cached():
    return hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY,
        host=HOPSWORKS_HOST,
    )

def get_hw_project_safe():
    """
    Fixes rare stale-client crash:
    AttributeError: _client._project_id
    """
    try:
        return _hw_project_cached()
    except Exception:
        try:
            _hw_project_cached.clear()
        except Exception:
            pass
        return hopsworks.login(
            project=HOPSWORKS_PROJECT,
            api_key_value=HOPSWORKS_API_KEY,
            host=HOPSWORKS_HOST,
        )

def normalize_preds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "timestamp_utc" not in df.columns:
        return pd.DataFrame()

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)

    if "event_time" not in df.columns:
        df["event_time"] = df["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")

    df["pk_time"] = df["timestamp_utc"].dt.tz_convert("Asia/Karachi")

    if "hazardous_alert" in df.columns and df["hazardous_alert"].dtype != bool:
        df["hazardous_alert"] = df["hazardous_alert"].astype(str).str.lower().isin(["true", "1", "yes"])
    elif "hazardous_alert" not in df.columns and "best_pred" in df.columns:
        df["hazardous_alert"] = pd.to_numeric(df["best_pred"], errors="coerce") >= float(HAZARDOUS_THRESHOLD)

    return df

def slice_horizon(df: pd.DataFrame, hours: int = 72) -> pd.DataFrame:
    """
    Force show ONLY next 72 hours from now.
    This prevents old rows staying on dashboard.
    """
    if df.empty or "timestamp_utc" not in df.columns:
        return df

    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    df = df.drop_duplicates(subset=["timestamp_utc"], keep="last").set_index("timestamp_utc")

    now = pd.Timestamp.now(tz="UTC")
    start = now.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)
    horizon = pd.date_range(start=start, periods=hours, freq="h", tz="UTC")

    out = df.reindex(horizon).reset_index().rename(columns={"index": "timestamp_utc"})
    out["pk_time"] = pd.to_datetime(out["timestamp_utc"], utc=True).dt.tz_convert("Asia/Karachi")
    return out

PRED_COLS = [
    "timestamp_utc","event_time","pk_time",
    "best_pred","best_model","best_model_rmse","best_reason","hazardous_alert",
    "pred_randomforest","pred_ridge","pred_neuralnet",
    "rmse_randomforest","rmse_ridge","rmse_neuralnet",
    "rf_version","ridge_version","mlp_version",
    "temp_c","humidity","pressure","wind_speed","city",
]

def get_or_create_pred_fg(fs):
    return fs.get_or_create_feature_group(
        name=PRED_FG_NAME,
        version=PRED_FG_VERSION,
        primary_key=["city", "event_time"],
        description="Next 3 days AQI predictions (3 models + best + RMSEs + alerts)",
        online_enabled=True,
    )

@st.cache_data(ttl=60, show_spinner=False)
def load_predictions_fast():
    """
    Fast load:
    - online first
    - slice to next 72 hours (always fresh)
    """
    project = get_hw_project_safe()
    fs = project.get_feature_store()
    fg = get_or_create_pred_fg(fs)

    df = pd.DataFrame()
    # ✅ ONLINE first (fast + latest)
    try:
        df = fg.read(online=True)
    except Exception:
        df = pd.DataFrame()

    # fallback to offline only if online empty
    if df is None or df.empty:
        try:
            df = fg.read()
        except Exception:
            return pd.DataFrame()

    df = normalize_preds(df)
    if df.empty:
        return df

    keep = [c for c in PRED_COLS if c in df.columns]
    df = df[keep].copy()
    df = slice_horizon(df, 72)
    return df

@st.cache_data(ttl=180, show_spinner=False)
def load_history_last_7_days():
    """
    Fix for "history only till 6 Feb":
    - read ONLINE first (latest)
    - filter by last 7 days from MAX timestamp in data
    """
    project = get_hw_project_safe()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    try:
        h = fg.read(online=True)
    except Exception:
        h = fg.read()

    h.columns = [c.lower() for c in h.columns]
    if "timestamp_utc" not in h.columns:
        return pd.DataFrame()

    h["timestamp_utc"] = pd.to_datetime(h["timestamp_utc"], utc=True, errors="coerce")
    h = h.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")

    if h.empty:
        return pd.DataFrame()

    end = h["timestamp_utc"].max()
    start = end - pd.Timedelta(days=7)
    h = h[h["timestamp_utc"] >= start].copy()
    h["pk_time"] = h["timestamp_utc"].dt.tz_convert("Asia/Karachi")

    keep = [c for c in ["pk_time","timestamp_utc","aqi","temp_c","humidity","pressure","wind_speed"] if c in h.columns]
    return h[keep].copy()

def build_export_report(df_pred: pd.DataFrame) -> pd.DataFrame:
    view = df_pred.copy()
    view = view.dropna(subset=["pk_time", "best_pred"]).copy()

    view["Date"] = view["pk_time"].dt.date.astype(str)
    view["Day"] = view["pk_time"].dt.strftime("%A")
    view["Time"] = view["pk_time"].dt.strftime("%H:%M:%S")
    view["AQI"] = pd.to_numeric(view["best_pred"], errors="coerce").round(0).astype("Int64")

    cats, recs = [], []
    for _, r in view.iterrows():
        _, em, pl, adv = aqi_category(r["best_pred"])
        cats.append(pl)
        recs.append(f"{em} {adv}")

    return pd.DataFrame({
        "Date": view["Date"],
        "Day": view["Day"],
        "Time": view["Time"],
        "AQI": view["AQI"].astype("Int64"),
        "Category": cats,
        "Type": ["Predicted"] * len(view),
        "Health_Recommendation": recs,
    })

@st.cache_data(ttl=180, show_spinner=False)
def export_cached(df_pred: pd.DataFrame) -> pd.DataFrame:
    return build_export_report(df_pred)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 🎛️ System Status")
    st.success("Model Active", icon="✅")
    st.markdown("---")

    st.markdown("## ⚙️ Actions")
    # ✅ Fast refresh: ONLY clears cache & re-reads latest predictions
    refresh_now = st.button("🔄 Refresh", use_container_width=True)
    auto_refresh = st.toggle("Auto refresh (120s)", value=False)
    show_raw = st.toggle("Show raw tables", value=False)

    st.markdown("---")
    st.markdown("## ⚠️ Threshold")
    st.write(f"Hazardous AQI: **{HAZARDOUS_THRESHOLD}**")

    st.markdown("---")
    st.markdown("## ✨ About")
    st.markdown(
        """
This dashboard delivers:

- **Live AQI monitoring** (Feature Store)  
- **72-hour forecast** (GitHub Actions daily inference)  
- **Automatic best-model selection** using **lowest RMSE**  
- **Export-ready reporting**  

<span class="small-note">Developed by <b>Wajiha Babar</b> • <b>2026</b></span>
""",
        unsafe_allow_html=True,
    )

if auto_refresh:
    st_autorefresh(interval=60_000, key="auto_refresh")

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
<div class="hero">
  <h1>🌫️ Air Quality Intelligence Dashboard</h1>
  <p>Live AQI + 72-hour forecast • Best-model selection • Health guidance • Exportable reports</p>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

# -----------------------------
# refresh behaviour
# -----------------------------
if refresh_now:
    st.cache_data.clear()
    st.rerun()

# -----------------------------
# Load Predictions 
# -----------------------------
df_pred = load_predictions_fast()

if df_pred.empty or ("best_pred" not in df_pred.columns):
    st.info("No predictions found yet. (Wait for GitHub Actions inference run) then click **Refresh Data (Fast)**.")
    st.stop()

# -----------------------------
# Current status card
# -----------------------------
first_valid_idx = df_pred["best_pred"].first_valid_index()
if first_valid_idx is None:
    st.error("Forecast rows exist but AQI values are missing. Click **Refresh Data (Fast)**.")
    st.stop()

current = df_pred.loc[first_valid_idx]
curr_aqi = float(pd.to_numeric(current.get("best_pred", np.nan), errors="coerce") or 0.0)
_, emoji, pill, advice = aqi_category(curr_aqi)

last_valid_idx = df_pred["best_pred"].last_valid_index()
latest = df_pred.loc[last_valid_idx] if last_valid_idx is not None else df_pred.iloc[-1]

best_model_now = str(latest.get("best_model", "—"))
best_rmse_now = latest.get("best_model_rmse", None)
best_reason = latest.get("best_reason", None) or "Selected automatically because it achieved the lowest RMSE on validation."

st.markdown(
    f"""
<div class="card" style="border-left:4px solid rgba(56,189,248,0.95);">
  <div class="card-title" style="font-size:1.05rem; font-weight:900;">
    Current Air Quality: <span class="kpi">{pill}</span> (AQI {int(round(curr_aqi))})
  </div>
  <div class="card-sub">{emoji} {advice}</div>
  <div class="card-sub" style="margin-top:10px;">
    <b>Best Model:</b> <span class="kpi">{best_model_now}</span> •
    <b>RMSE:</b> <span class="kpi">{fmt(best_rmse_now, 4)}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

# -----------------------------
# Key Metrics (daily mean)
# -----------------------------
tmp = df_pred.dropna(subset=["pk_time", "best_pred"]).copy()
tmp["date_pk"] = tmp["pk_time"].dt.date
daily = tmp.groupby("date_pk", as_index=False)["best_pred"].mean().sort_values("date_pk").head(4)

st.markdown("## 📊 Key Metrics")
c1, c2, c3, c4 = st.columns(4)
cols = [c1, c2, c3, c4]
labels = ["TODAY'S AQI", "TOMORROW", "DAY +2", "DAY +3"]

for i in range(4):
    val = float(daily["best_pred"].iloc[i]) if len(daily) > i else None
    _, em, pl, _ = aqi_category(val)
    with cols[i]:
        st.markdown(
            f"""<div class="card">
            <div class="card-title">{labels[i]}</div>
            <div class="card-value">{fmt(val,0)}</div>
            <div class="card-sub"><span class="pill">{em} {pl}</span></div>
            </div>""",
            unsafe_allow_html=True,
        )

st.divider()

# -----------------------------
# Trend chart (Observed + Best)
# -----------------------------
st.markdown("## 📈 Air Quality Index — Historical & Forecast Trend (Clean)")

hist = load_history_last_7_days()
if not hist.empty and "pk_time" in hist.columns:
    hist = hist.sort_values("pk_time").copy()

if not hist.empty and "aqi" in hist.columns:
    hist["aqi"] = pd.to_numeric(hist["aqi"], errors="coerce").clip(lower=0, upper=500)

pred = df_pred.copy()
pred["best_pred"] = pd.to_numeric(pred["best_pred"], errors="coerce").clip(lower=0, upper=500)
pred = pred.dropna(subset=["pk_time"]).sort_values("pk_time").reset_index(drop=True)

pred_clean = pred.dropna(subset=["best_pred"]).copy()
forecast_start = pred_clean["pk_time"].min()
forecast_end = pred_clean["pk_time"].max()

x_min = hist["pk_time"].min() if (not hist.empty and "pk_time" in hist.columns) else (forecast_start - pd.Timedelta(days=2))
x_max = forecast_end + pd.Timedelta(hours=1)

fig = go.Figure()

bands = [
    (0, 50,  "rgba(34,197,94,0.10)"),
    (50, 100,"rgba(234,179,8,0.10)"),
    (100,150,"rgba(249,115,22,0.10)"),
    (150,200,"rgba(239,68,68,0.10)"),
    (200,300,"rgba(99,102,241,0.08)"),
    (300,500,"rgba(120,113,108,0.08)"),
]
for lo, hi, color in bands:
    fig.add_hrect(y0=lo, y1=hi, fillcolor=color, line_width=0)

if not hist.empty and "aqi" in hist.columns:
    h = hist.dropna(subset=["pk_time", "aqi"]).copy()
    fig.add_trace(go.Scatter(
        x=h["pk_time"], y=h["aqi"],
        mode="lines+markers", name="Observed",
        line=dict(width=3.0),
        marker=dict(size=5, symbol="circle"),
        hovertemplate="<b>%{x|%b %d, %I:%M %p}</b><br>AQI: <b>%{y:.0f}</b><extra></extra>",
    ))

fig.add_vrect(x0=forecast_start, x1=forecast_end, fillcolor="rgba(59,130,246,0.10)", line_width=0)
fig.add_vline(x=forecast_start, line_width=2, line_dash="dash", opacity=0.9)

fig.add_trace(go.Scatter(
    x=pred_clean["pk_time"], y=pred_clean["best_pred"],
    mode="lines+markers", name="Predicted (Best)",
    line=dict(width=3.4),
    marker=dict(size=6, symbol="diamond"),
    hovertemplate="<b>%{x|%b %d, %I:%M %p}</b><br>Forecast AQI: <b>%{y:.0f}</b><extra></extra>",
))

y_vals = []
if not hist.empty and "aqi" in hist.columns:
    y_vals += list(hist["aqi"].dropna().astype(float).values)
y_vals += list(pred_clean["best_pred"].dropna().astype(float).values)

y_arr = np.array(y_vals, dtype="float")
y_arr = y_arr[np.isfinite(y_arr)]
if len(y_arr) > 0:
    q05 = float(np.quantile(y_arr, 0.05))
    q95 = float(np.quantile(y_arr, 0.95))
    pad = max(10.0, (q95 - q05) * 0.30)
    y0 = max(0.0, q05 - pad)
    y1 = min(300.0, q95 + pad)
else:
    y0, y1 = 0, 250

fig.update_layout(
    height=620,
    margin=dict(l=14, r=14, t=40, b=10),
    template="plotly_white",
    paper_bgcolor="rgba(255,255,255,0.03)",
    plot_bgcolor="rgba(255,255,255,0.96)",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.12,
        xanchor="right",
        x=1,
        bgcolor="rgba(255,255,255,0.65)",
        bordercolor="rgba(0,0,0,0.12)",
        borderwidth=1,
    ),
)
fig.update_xaxes(title="Time (PKT)", showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False, range=[x_min, x_max])
fig.update_yaxes(title="AQI", showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False, range=[y0, y1])

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 3-model comparison chart
# -----------------------------
st.markdown("## 📉 3-Model Comparison — 72-hour Forecast (All Models)")

m = df_pred.copy().dropna(subset=["pk_time"]).sort_values("pk_time")
for c in ["pred_randomforest", "pred_ridge", "pred_neuralnet", "best_pred"]:
    if c in m.columns:
        m[c] = pd.to_numeric(m[c], errors="coerce").clip(lower=0, upper=500)

fig2 = go.Figure()
if "pred_randomforest" in m.columns:
    fig2.add_trace(go.Scatter(x=m["pk_time"], y=m["pred_randomforest"], mode="lines", name="RandomForest"))
if "pred_ridge" in m.columns:
    fig2.add_trace(go.Scatter(x=m["pk_time"], y=m["pred_ridge"], mode="lines", name="Ridge"))
if "pred_neuralnet" in m.columns:
    fig2.add_trace(go.Scatter(x=m["pk_time"], y=m["pred_neuralnet"], mode="lines", name="NeuralNet (MLP)"))
if "best_pred" in m.columns:
    fig2.add_trace(go.Scatter(x=m["pk_time"], y=m["best_pred"], mode="lines", name="Best Selected", line=dict(width=4)))

fig2.update_layout(
    height=520,
    template="plotly_white",
    paper_bgcolor="rgba(255,255,255,0.03)",
    plot_bgcolor="rgba(255,255,255,0.96)",
    margin=dict(l=14, r=14, t=40, b=10),
    hovermode="x unified",
    legend=dict(orientation="h", y=1.12, x=1, xanchor="right",
                bgcolor="rgba(255,255,255,0.65)", bordercolor="rgba(0,0,0,0.12)", borderwidth=1),
)
fig2.update_xaxes(title="Time (PKT)", gridcolor="rgba(0,0,0,0.08)")
fig2.update_yaxes(title="AQI", gridcolor="rgba(0,0,0,0.08)")
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Tabs (stable)
# -----------------------------
tab = st.radio(
    "Navigation",
    ["🧾 Forecast Table", "⬇️ Export Report", "🩺 Health Guidance", "📌 Data Insights", "📚 Historical Overview"],
    horizontal=True,
    label_visibility="collapsed",
    key="tab",
)

if tab == "🧾 Forecast Table":
    st.markdown("## Complete Forecast Report")
    report = export_cached(df_pred)
    st.dataframe(report, use_container_width=True, hide_index=True)

    st.write("")
    st.markdown('<div class="big-btn">', unsafe_allow_html=True)
    st.download_button(
        "⬇️ Download Forecast Report (CSV)",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="aqi_forecast_report.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif tab == "⬇️ Export Report":
    st.markdown("## Export (CSV)")
    st.markdown(
        """<div class="card">
        <div class="card-title">One-click export</div>
        <div class="card-sub">
        Download a clean, shareable CSV of the 72-hour forecast — date-wise values, AQI category, and health guidance included.
        </div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.write("")

    report = export_cached(df_pred)
    st.markdown('<div class="big-btn">', unsafe_allow_html=True)
    st.download_button(
        "⬇️ Download Forecast Report (CSV)",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="aqi_forecast_report.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif tab == "🩺 Health Guidance":
    st.markdown("## Personalized Health Recommendations")
    tmp2 = df_pred.dropna(subset=["pk_time", "best_pred"]).copy()
    tmp2["date_pk"] = tmp2["pk_time"].dt.date
    daily2 = tmp2.groupby("date_pk", as_index=False)["best_pred"].mean().sort_values("date_pk").head(7)

    for i in range(len(daily2)):
        aqi_val = float(daily2["best_pred"].iloc[i])
        dt = pd.to_datetime(str(daily2["date_pk"].iloc[i]))
        _, em, pl, adv = aqi_category(aqi_val)
        st.markdown(
            f"""<div class="card" style="border-left:4px solid rgba(56,189,248,0.95);">
            <div class="card-title" style="font-size:1.05rem; font-weight:900;">{dt.strftime("%A, %B %d")}</div>
            <div class="card-sub"><b>Forecast:</b> <span class="kpi">{pl}</span> (AQI {int(round(aqi_val))})</div>
            <div class="card-sub">{em} {adv}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.write("")

elif tab == "📌 Data Insights":
    st.markdown("## Model & Forecast Insights")

    rmse_rf = latest.get("rmse_randomforest", None)
    rmse_ridge = latest.get("rmse_ridge", None)
    rmse_mlp = latest.get("rmse_neuralnet", None)

    rf_v = latest.get("rf_version", "—")
    ridge_v = latest.get("ridge_version", "—")
    mlp_v = latest.get("mlp_version", "—")

    a, b = st.columns(2)
    with a:
        st.markdown(
            f"""<div class="card">
            <div class="card-title">Best Model Selection</div>
            <div class="card-value">{best_model_now}</div>
            <div class="card-sub">{best_reason}</div>
            <div class="card-sub">Best RMSE: <b class="kpi">{fmt(best_rmse_now, 4)}</b></div>
            </div>""",
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            f"""<div class="card">
            <div class="card-title">Models (versions + RMSE)</div>
            <div class="card-sub">• RandomForest: v<b>{rf_v}</b> • RMSE <b>{fmt(rmse_rf,4)}</b></div>
            <div class="card-sub">• Ridge: v<b>{ridge_v}</b> • RMSE <b>{fmt(rmse_ridge,4)}</b></div>
            <div class="card-sub">• NeuralNet: v<b>{mlp_v}</b> • RMSE <b>{fmt(rmse_mlp,4)}</b></div>
            <div class="card-sub">Dashboard shows <b>Best-selected</b> forecast (+ observed history).</div>
            </div>""",
            unsafe_allow_html=True,
        )

elif tab == "📚 Historical Overview":
    st.markdown("## Environmental Parameters — Historical Overview (Past 7 Days)")
    hist2 = load_history_last_7_days()

    if hist2.empty or "aqi" not in hist2.columns:
        st.info("History not available in aqi_features_v2 (or missing columns).")
    else:
        cols = st.columns(2)

        def plot_series(colname, title, slot):
            if colname not in hist2.columns:
                with slot:
                    st.warning(f"{title}: not available")
                return
            d = hist2.dropna(subset=["pk_time", colname])
            figx = go.Figure()
            figx.add_trace(go.Scatter(x=d["pk_time"], y=d[colname], mode="lines", name=title))
            figx.update_layout(
                height=320,
                template="plotly_white",
                paper_bgcolor="rgba(255,255,255,0.03)",
                plot_bgcolor="rgba(255,255,255,0.94)",
                margin=dict(l=10, r=10, t=40, b=10),
                title=title,
            )
            figx.update_xaxes(gridcolor="rgba(0,0,0,0.08)")
            figx.update_yaxes(gridcolor="rgba(0,0,0,0.08)")
            with slot:
                st.plotly_chart(figx, use_container_width=True)

        plot_series("aqi", "Observed AQI (Last 7 Days)", cols[0])
        plot_series("temp_c", "Temperature (°C)", cols[1])

        cols2 = st.columns(2)
        plot_series("humidity", "Humidity (%)", cols2[0])
        plot_series("wind_speed", "Wind Speed (m/s)", cols2[1])

if show_raw:
    st.write("")
    st.markdown("### Raw Predictions (debug)")
    st.dataframe(df_pred, use_container_width=True, hide_index=True)
