from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import joblib
import hopsworks

from dotenv import load_dotenv

# ----------------------------
# Force-load .env from root
# ----------------------------
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)


# ---------------------------------
# SETTINGS
# ---------------------------------
FEATURE_GROUP_NAME = "aqi_features_v2"
FEATURE_GROUP_VERSION = 1

MODEL_NAME = "aqi_random_forest"

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_PNG = OUTPUT_DIR / "shap_summary.png"


# ---------------------------------
# ENV
# ---------------------------------
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "").strip()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "").strip()
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "").strip()


def die(msg: str):
    print(f"\n❌ {msg}\n")
    sys.exit(1)


def require_pkg():
    try:
        import shap  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
        return True
    except Exception:
        print("\n❌ Missing packages for SHAP plotting.")
        print("✅ Run:\n   pip install shap matplotlib\n")
        return False


def normalize_host_to_domain(raw_host: str) -> str:
    """Return ONLY the domain for hopsworks.login(host=...)."""
    if not raw_host:
        return ""
    host = raw_host.strip()
    host = host.replace("https://https://", "https://").replace("http://http://", "http://")
    host = host.replace(":443:443", ":443")

    if host.startswith("https://"):
        host = host[len("https://"):]
    elif host.startswith("http://"):
        host = host[len("http://"):]

    host = host.split("/")[0].strip()
    if ":" in host:
        host = host.split(":", 1)[0]
    return host


def find_model_file(model_dir: Path) -> Optional[Path]:
    files = list(model_dir.rglob("*.joblib")) + list(model_dir.rglob("*.pkl"))
    return files[0] if files else None


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df = df.sort_values("timestamp_utc")

    if "aqi" not in df.columns:
        die("Feature group me 'aqi' column missing hai.")

    df["aqi_lag1"] = df["aqi"].shift(1)
    df["aqi_lag24"] = df["aqi"].shift(24)
    df["aqi_roll24"] = df["aqi"].rolling(24).mean()
    df["aqi_roll7d"] = df["aqi"].rolling(24 * 7).mean()
    df["aqi_diff1"] = df["aqi"] - df["aqi_lag1"]

    df = df.dropna().copy()
    return df


def main():
    if not require_pkg():
        return

    import shap
    import matplotlib.pyplot as plt

    # show env values (SAFE)
    print("✅ .env loaded from:", ENV_PATH)
    print("✅ HOPSWORKS_PROJECT:", HOPSWORKS_PROJECT)
    print("✅ HOPSWORKS_HOST:", HOPSWORKS_HOST)
    print("✅ API KEY present:", "YES" if bool(HOPSWORKS_API_KEY) else "NO")

    if not HOPSWORKS_PROJECT or not HOPSWORKS_API_KEY or not HOPSWORKS_HOST:
        die("Missing .env values. Check HOPSWORKS_PROJECT/HOST/API_KEY in .env")

    host_domain = normalize_host_to_domain(HOPSWORKS_HOST)

    if "c.app.hopsworks.ai" in host_domain:
        die("Wrong host detected (c.app...). Use: eu-west.cloud.hopsworks.ai")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n✅ Logging in (no prompt)...")
    print("✅ Host domain:", host_domain)

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY,
        host=host_domain,
    )

    fs = project.get_feature_store()
    mr = project.get_model_registry()

    print("\n✅ Reading feature group (ArrowFlight, no Hive)...")
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION)
    df = fg.read(dataframe_type="pandas", read_options={"use_hive": False})

    if df is None or df.empty:
        die("Feature group read empty. Data nahi aa raha.")

    print("✅ Building features...")
    df = build_features(df)

    X = df.drop(columns=["aqi", "timestamp_utc", "event_time", "city"], errors="ignore")
    X_sample = X.tail(400).copy()
    if X_sample.empty:
        die("X_sample empty ho gaya. Feature group me data kam hai.")

    print("\n✅ Downloading model from registry (latest)...")
    models = mr.get_models(MODEL_NAME)
    if not models:
        die(f"Model not found in registry: {MODEL_NAME}")

    latest = sorted(models, key=lambda m: m.version)[-1]
    meta = mr.get_model(MODEL_NAME, version=latest.version)
    model_dir = Path(meta.download())

    model_file = find_model_file(model_dir)
    if not model_file:
        die(f"Downloaded model folder me .joblib/.pkl nahi mila: {model_dir}")

    print("✅ Loading model:", model_file)
    model = joblib.load(model_file)

    print("\n✅ Computing SHAP...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    print("✅ Saving plot:", OUTPUT_PNG)
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200)
    plt.close()

    print(f"\n✅ DONE! SHAP saved: {OUTPUT_PNG}\n")


if __name__ == "__main__":
    main()
