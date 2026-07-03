# 📦 Predictive Inventory ML Model — Next-Day Reservation Forecasting

> Proactive warehouse relocation using 3-signal historical pattern analysis — inspired by work at Flipkart Grocery (Walmart Group) that reduced reactive manual warehouse movements by **8%**.

---

## 🎯 Business Problem

In a grocery warehouse, items sit in **deep storage** until orders arrive. When demand spikes unexpectedly, staff must manually relocate items to **forward picking zones** — a reactive, slow, and expensive process.

**This model predicts next-day reservations per SKU so the warehouse can relocate proactively — before demand hits.**

---

## 💡 The 3-Signal Forecasting Logic

The model is built around **3 core historical signals** that each capture a different demand pattern:

| Signal | Lookback | Pattern it captures | Correlation with actual |
|---|---|---|---|
| **S1 — Avg last 7 days** | 7 days | Current demand trend | r = 0.937 |
| **S2 — Same day last month** | 28 days (same weekday) | Monthly / end-of-month cycle | r = 0.906 |
| **S3 — Same day last year** | 364 days (same weekday) | Annual seasonality (festive, seasonal) | r = 0.953 |
| **Weighted blend** | All 3 combined | All patterns together | r = 0.954 |

These signals are **explicit and auditable** — a warehouse manager can see exactly why a forecast was made.

---

## 📅 Why all 3 signals matter

![Seasonality](outputs/charts/03_seasonality.png)

- **Weekly**: Snacks and Beverages spike on weekends; Dairy and Staples peak on weekdays
- **Monthly**: End-of-month restocking creates a consistent surge on days 27–31
- **Yearly**: Nov–Dec festive season drives a 30–60% uplift; Feb–Mar sees a summer lull

---

## 🔍 Signals in action — actual vs forecast

![Signals Story](outputs/charts/01_signals_story.png)

The chart shows all 3 signals alongside the actual order line — you can see Signal 3 (yearly) picking up the festive surge that the 7-day average hasn't caught yet.

---

## 📊 Model Results

Three approaches were compared on a **90-day held-out test set** (data the model never saw during training):

| Model | MAE | MAPE | R² |
|---|---|---|---|
| Weighted Baseline (S1×0.5 + S2×0.3 + S3×0.2) | 16.45 | 11.40% | 0.899 |
| Optimised Weights (S1×0.4 + S2×0.2 + S3×0.4) | 15.98 | 11.14% | 0.905 |
| **Random Forest ✅** | **13.70** | **9.32%** | **0.923** |

![Model Comparison](outputs/charts/04_model_comparison.png)

The Random Forest uses the **3 signals as its primary features**, then learns when calendar context (day of week, month-end flags) should adjust the signal blend.

---

## 🔍 What the model learned

![Feature Importance](outputs/charts/02_feature_importance.png)

**Signal 3 (same day last year) dominates** — the model discovered that annual patterns are the strongest predictor for grocery items, consistent with the strong festive and seasonal cycles in the data.

---

## 🚦 Next-Day Relocation Output

Each SKU is scored against its recent average to determine warehouse action:

| Priority | Demand ratio | Action |
|---|---|---|
| 🔴 HIGH | ≥ 1.30× recent avg | Relocate to forward picking zone immediately |
| 🟡 MEDIUM | 1.10×–1.30× | Stage for relocation, monitor |
| 🟢 LOW | < 1.10× | No action needed |

![Relocation Priority](outputs/charts/05_relocation_priority.png)

Sample output from `outputs/next_day_forecast.csv`:

```
SKU       Category  S1: Avg 7d  S2: Last Month  S3: Last Year  Baseline  ML Forecast  Priority
DAI-001   Dairy        166.4       162.0          153.0           160        170       🟢 LOW
SNA-001   Snacks       178.7       120.0           73.0           125         85       🟢 LOW
STA-001   Staples      284.4       245.0          211.0           247        228       🟢 LOW
```

---

## 📁 Project Structure

```
inventory-ml-model/
├── data/
│   ├── orders.csv              # 3 years × 15 SKUs daily order history (16,440 rows)
│   └── features.csv            # Engineered 3-signal feature set (10,950 rows)
├── src/
│   ├── generate_data.py        # Synthetic data with real weekly/monthly/yearly patterns
│   ├── features.py             # Build the 3 core signals + calendar features
│   ├── train.py                # Compare 3 modelling approaches; save best
│   ├── predict.py              # Generate next-day forecast with signal breakdown
│   └── analyse.py              # All 5 charts
├── outputs/
│   ├── model.pkl               # Trained Random Forest
│   ├── model_results.csv       # Model comparison table
│   ├── signal_weights.json     # Optimised signal weights
│   ├── feature_importance.csv  # Signal importance scores
│   ├── test_predictions.csv    # Actual vs predicted (90-day test)
│   ├── next_day_forecast.csv   # Next-day forecast per SKU
│   └── charts/                 # All 5 visualisations
├── requirements.txt
└── README.md
```

---

## 🚀 Run It Yourself

```bash
git clone https://github.com/Pranavks27/inventory-ml-model.git
cd inventory-ml-model
pip install -r requirements.txt

python src/generate_data.py   # Generate 3 years of order history
python src/features.py        # Build the 3 core signals
python src/train.py           # Train & compare models
python src/predict.py         # Forecast next-day reservations
python src/analyse.py         # Generate all charts
```

---

## 🛠️ Tech Stack

`Python 3.11` · `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `joblib`

---

## 🏭 Real-World Impact

This model replicates the logic from a production system built at **Flipkart Grocery (Walmart Group)**:
- Analysed reservation patterns across all warehouse SKUs daily
- Enabled warehouse teams to pre-position inventory **before** demand spikes
- Replaced reactive, gut-feel decisions with data-driven relocation scheduling
- Reduced reactive manual warehouse movements by **8%**

---

## 👤 Author

**Pranav KS** — Data Analytics Professional | Master of Data Science @ RMIT University  
📧 kspranav2000@gmail.com · [LinkedIn](https://linkedin.com/in/pranav-ks) · [GitHub](https://github.com/Pranavks27)
