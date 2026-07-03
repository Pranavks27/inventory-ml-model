"""
train.py — Train and compare forecasting approaches.

Stage 1 — Weighted Signal Baseline:
    forecast = 0.50 × Signal1 + 0.30 × Signal2 + 0.20 × Signal3

Stage 2 — Optimised Weights (find the best blend of the 3 signals):
    Grid search over weight combinations

Stage 3 — Random Forest (learns residual corrections on the 3 signals + calendar):
    Uses the 3 signals AS features, so their importance is directly interpretable

Outputs:
    outputs/model.pkl
    outputs/model_results.csv
    outputs/signal_weights.json
    outputs/feature_importance.csv
    outputs/test_predictions.csv
"""

import json
import joblib
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

Path("outputs").mkdir(exist_ok=True)

# Features the RF uses — the 3 signals PLUS calendar context
RF_FEATURES = [
    "s1_last_7d_avg",
    "s2_same_day_lmonth",
    "s3_same_day_lyear",
    "rolling_std_7d",
    "trend_slope_7d",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "is_month_end",
    "is_festive",
    "category_enc",
]
TARGET = "actual_next_day"


def metrics(name, y_true, y_pred):
    mape_val = np.mean(np.abs((y_true - y_pred) / y_true.replace(0, 1))) * 100
    return {
        "model": name,
        "MAE":   round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE":  round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "MAPE":  round(mape_val, 2),
        "R2":    round(r2_score(y_true, y_pred), 4),
    }


def train():
    print("[+] Loading features...")
    df = pd.read_csv("data/features.csv", parse_dates=["date"])
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)

    # Label encode category
    le = LabelEncoder()
    df["category_enc"] = le.fit_transform(df["category"])

    # Time-based split — last 90 days as test (never leaked into training)
    cutoff   = df["date"].max() - pd.Timedelta(days=90)
    train_df = df[df["date"] <= cutoff].copy()
    test_df  = df[df["date"] >  cutoff].copy()
    print(f"    Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")

    y_train = train_df[TARGET]
    y_test  = test_df[TARGET]
    results = []

    # ── Stage 1: Fixed weighted baseline (0.5 / 0.3 / 0.2) ──────────────────
    print("\n[1] Fixed weighted baseline (50% S1 + 30% S2 + 20% S3)...")
    pred_fixed = (0.50 * test_df["s1_last_7d_avg"] +
                  0.30 * test_df["s2_same_day_lmonth"] +
                  0.20 * test_df["s3_same_day_lyear"])
    results.append(metrics("Weighted Baseline (fixed)", y_test, pred_fixed))
    print(f"    {results[-1]}")

    # ── Stage 2: Optimised signal weights ────────────────────────────────────
    print("\n[2] Optimising signal weights on training set...")
    best_mae, best_w = 9999, (0.5, 0.3, 0.2)
    step = 0.1
    candidates = [(round(w1,1), round(w2,1), round(w3,1))
                  for w1 in np.arange(0.3, 0.8, step)
                  for w2 in np.arange(0.1, 0.5, step)
                  for w3 in np.arange(0.1, 0.4, step)
                  if abs(w1 + w2 + w3 - 1.0) < 0.01]

    for w1, w2, w3 in candidates:
        pred = (w1 * train_df["s1_last_7d_avg"] +
                w2 * train_df["s2_same_day_lmonth"] +
                w3 * train_df["s3_same_day_lyear"])
        mae = mean_absolute_error(y_train, pred)
        if mae < best_mae:
            best_mae, best_w = mae, (w1, w2, w3)

    w1, w2, w3 = best_w
    pred_opt = (w1 * test_df["s1_last_7d_avg"] +
                w2 * test_df["s2_same_day_lmonth"] +
                w3 * test_df["s3_same_day_lyear"])
    results.append(metrics(f"Optimised Weights ({w1}/{w2}/{w3})", y_test, pred_opt))
    print(f"    Best weights: S1={w1}, S2={w2}, S3={w3}")
    print(f"    {results[-1]}")

    with open("outputs/signal_weights.json", "w") as f:
        json.dump({"s1_last_7d_avg": w1, "s2_same_day_lmonth": w2,
                   "s3_same_day_lyear": w3}, f, indent=2)

    # ── Stage 3: Random Forest using 3 signals as features ───────────────────
    print("\n[3] Random Forest (3 signals + calendar features)...")
    X_train = train_df[RF_FEATURES]
    X_test  = test_df[RF_FEATURES]

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    pred_rf = pd.Series(rf.predict(X_test), index=y_test.index)
    results.append(metrics("Random Forest", y_test, pred_rf))
    print(f"    {results[-1]}")

    # ── Feature importance — signals vs calendar ──────────────────────────────
    fi = pd.DataFrame({
        "feature":    RF_FEATURES,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv("outputs/feature_importance.csv", index=False)

    # ── Save results ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/model_results.csv", index=False)

    # Best model = RF
    joblib.dump({
        "model":         rf,
        "label_encoder": le,
        "features":      RF_FEATURES,
        "opt_weights":   {"s1": w1, "s2": w2, "s3": w3},
    }, "outputs/model.pkl")

    # Test predictions with signal breakdown
    test_out = test_df[["date","predict_date","sku_id","category",
                         "s1_last_7d_avg","s2_same_day_lmonth","s3_same_day_lyear",
                         "baseline_forecast","actual_next_day"]].copy()
    test_out["rf_predicted"] = pred_rf.values.round().astype(int)
    test_out.to_csv("outputs/test_predictions.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(results_df.to_string(index=False))
    print("\nFEATURE IMPORTANCE (top 5):")
    print(fi.head(5).to_string(index=False))
    print("="*60)

    best = results_df.loc[results_df["MAE"].idxmin()]
    print(f"\n[✓] Best: {best['model']}  MAE={best['MAE']}  MAPE={best['MAPE']}%  R²={best['R2']}")

    return results_df, fi


if __name__ == "__main__":
    train()
