"""
predict.py — Next-day reservation forecast with full signal breakdown per SKU.

For each SKU shows:
  - Signal 1: avg last 7 days
  - Signal 2: same day last month
  - Signal 3: same day last year
  - Weighted blend (baseline)
  - ML-refined prediction
  - Relocation priority

Run:
    python src/predict.py
    python src/predict.py --date 2024-12-30
"""

import argparse
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from features import build_features

Path("outputs").mkdir(exist_ok=True)


PRIORITY_RULES = [
    (1.30, "🔴  HIGH    — relocate to forward picking zone NOW"),
    (1.10, "🟡  MEDIUM  — stage for relocation, monitor"),
    (0.00, "🟢  LOW     — no action needed"),
]

def get_priority(predicted, signal1_avg):
    ratio = predicted / signal1_avg if signal1_avg > 0 else 1.0
    for threshold, label in PRIORITY_RULES:
        if ratio >= threshold:
            return label, round(ratio, 2)
    return PRIORITY_RULES[-1][1], round(ratio, 2)


def predict(target_date: str = None):
    print("[+] Loading model...")
    bundle  = joblib.load("outputs/model.pkl")
    model   = bundle["model"]
    le      = bundle["label_encoder"]
    feats   = bundle["features"]
    weights = bundle["opt_weights"]

    print("[+] Building features from order history...")
    raw  = pd.read_csv("data/orders.csv")
    feat = build_features(raw)
    feat["date"] = pd.to_datetime(feat["date"])

    # Use the most recent available date per SKU
    latest = feat.groupby("sku_id").last().reset_index()
    predict_for = pd.to_datetime(target_date) + pd.Timedelta(days=1) if target_date \
                  else (feat["date"].max() + pd.Timedelta(days=1))

    print(f"[+] Forecasting reservations for: {predict_for.date()}\n")

    latest["category_enc"] = le.transform(latest["category"])
    X = latest[feats]
    latest["ml_predicted"] = model.predict(X).round().astype(int).clip(min=1)

    # Weighted baseline using optimised weights
    latest["baseline"] = (
        weights["s1"] * latest["s1_last_7d_avg"] +
        weights["s2"] * latest["s2_same_day_lmonth"] +
        weights["s3"] * latest["s3_same_day_lyear"]
    ).round().astype(int)

    # Relocation priority based on ML prediction vs recent average
    priorities, ratios = zip(*latest.apply(
        lambda r: get_priority(r["ml_predicted"], r["s1_last_7d_avg"]), axis=1
    ))
    latest["relocation_priority"] = priorities
    latest["demand_ratio"]        = ratios

    # ── Output table ──────────────────────────────────────────────────────────
    out = latest[[
        "sku_id", "category",
        "s1_last_7d_avg",
        "s2_same_day_lmonth",
        "s3_same_day_lyear",
        "baseline",
        "ml_predicted",
        "relocation_priority",
        "demand_ratio",
    ]].sort_values("ml_predicted", ascending=False).reset_index(drop=True)

    out.columns = [
        "SKU", "Category",
        "S1: Avg Last 7d",
        "S2: Same Day Last Month",
        "S3: Same Day Last Year",
        "Baseline Forecast",
        "ML Forecast",
        "Relocation Priority",
        "Demand Ratio",
    ]

    out.to_csv("outputs/next_day_forecast.csv", index=False)

    # ── Print ─────────────────────────────────────────────────────────────────
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(out.to_string(index=False))

    high   = out["Relocation Priority"].str.contains("HIGH").sum()
    medium = out["Relocation Priority"].str.contains("MEDIUM").sum()
    low    = out["Relocation Priority"].str.contains("LOW").sum()

    print(f"\n{'='*60}")
    print(f"  Forecast date : {predict_for.date()}")
    print(f"  Signal weights: S1={weights['s1']} | S2={weights['s2']} | S3={weights['s3']}")
    print(f"  🔴 HIGH   : {high}  SKUs — immediate relocation needed")
    print(f"  🟡 MEDIUM : {medium}  SKUs — stage for relocation")
    print(f"  🟢 LOW    : {low}  SKUs — no action")
    print(f"{'='*60}")
    print(f"\n[✓] Saved → outputs/next_day_forecast.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()
    predict(args.date)
