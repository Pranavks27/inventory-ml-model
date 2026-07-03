"""
features.py — Build the 3 core forecasting signals per SKU per day.

Core signals (the business logic):
  Signal 1 — last_7d_avg        : average orders over the past 7 days  (recent trend)
  Signal 2 — same_day_last_month: orders on same weekday 4 weeks ago   (monthly pattern)
  Signal 3 — same_day_last_year : orders on same weekday 52 weeks ago  (yearly pattern)

These 3 signals are kept as explicit columns so their contribution stays transparent.
The weighted blend of them becomes the baseline forecast.
ML then learns residual corrections on top.
"""

import pandas as pd
import numpy as np
from pathlib import Path

Path("data").mkdir(exist_ok=True)

# How many days back for each signal
WEEKS_BACK_MONTH = 4    # 28 days  — same weekday last month
WEEKS_BACK_YEAR  = 52   # 364 days — same weekday last year
MIN_HISTORY_DAYS = 365  # need at least 1 year of history to use all 3 signals


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)

    records = []

    for sku_id, grp in df.groupby("sku_id"):
        grp    = grp.set_index("date").sort_index()
        orders = grp["orders"]
        dates  = orders.index
        cat    = grp["category"].iloc[0]

        for i in range(MIN_HISTORY_DAYS, len(orders) - 1):
            today    = dates[i]
            tomorrow = dates[i + 1]

            # ── Signal 1: Average of last 7 days ─────────────────────────────
            s1_last7 = orders.iloc[i - 7 : i].mean()

            # ── Signal 2: Same weekday 4 weeks ago ───────────────────────────
            s2_idx   = i - (WEEKS_BACK_MONTH * 7)
            s2_sdlm  = float(orders.iloc[s2_idx]) if s2_idx >= 0 else s1_last7

            # ── Signal 3: Same weekday 52 weeks ago ──────────────────────────
            s3_idx   = i - (WEEKS_BACK_YEAR * 7)
            s3_sdly  = float(orders.iloc[s3_idx]) if s3_idx >= 0 else s1_last7

            # ── Weighted blend (business baseline) ───────────────────────────
            # 50% recent trend | 30% monthly pattern | 20% yearly pattern
            baseline = 0.50 * s1_last7 + 0.30 * s2_sdlm + 0.20 * s3_sdly

            # ── Residual features for ML to correct ──────────────────────────
            rolling_std = orders.iloc[i - 7 : i].std()
            trend_slope = (orders.iloc[i] - orders.iloc[i - 7]) / 7  # 7-day slope

            records.append({
                # Identifiers
                "date":               today,
                "predict_date":       tomorrow,
                "sku_id":             sku_id,
                "category":           cat,

                # ── The 3 core signals ───────────────────────────────────────
                "s1_last_7d_avg":     round(s1_last7, 2),
                "s2_same_day_lmonth": round(s2_sdlm,  2),
                "s3_same_day_lyear":  round(s3_sdly,  2),

                # Weighted baseline (explicit, auditable)
                "baseline_forecast":  round(baseline,  2),

                # ML correction features
                "rolling_std_7d":     round(rolling_std, 2),
                "trend_slope_7d":     round(trend_slope, 2),
                "day_of_week":        today.dayofweek,
                "day_of_month":       today.day,
                "month":              today.month,
                "is_weekend":         int(today.dayofweek >= 5),
                "is_month_end":       int(today.day >= 27),
                "is_festive":         int(today.month in [11, 12]),

                # Target
                "actual_next_day":    int(orders.iloc[i + 1]),
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    raw  = pd.read_csv("data/orders.csv")
    feat = build_features(raw)
    feat.to_csv("data/features.csv", index=False)

    print(f"[✓] Features: {feat.shape[0]:,} rows × {feat.shape[1]} cols")
    print(f"\nSample — 3 core signals vs actual:")
    sample = feat[["date","sku_id","s1_last_7d_avg","s2_same_day_lmonth",
                   "s3_same_day_lyear","baseline_forecast","actual_next_day"]].head(8)
    print(sample.to_string(index=False))

    # Show signal correlations with actual
    print("\nCorrelation of each signal with actual next-day orders:")
    for col in ["s1_last_7d_avg","s2_same_day_lmonth","s3_same_day_lyear","baseline_forecast"]:
        r = feat[col].corr(feat["actual_next_day"])
        print(f"  {col:30s}: r = {r:.4f}")
