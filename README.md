# 📈 Multi-Horizon Sales Forecaster

> Demand forecasting with uncertainty quantification and what-if scenario analysis — built on the M5 Walmart public dataset.

## The Problem

Most forecasting tools produce a single number. Real procurement and promotional decisions require:
- A **range of outcomes**, not a point estimate
- The ability to ask **"what if I run a 20% promo?"** before committing budget
- Evidence the model works — not just a pretty forecast line

This app addresses all three.

---

## Features

| Tab | What it shows |
|---|---|
| 📊 Data Explorer | Sales history with promotion periods highlighted |
| 🔮 Forecast | 1/4/12-week ahead forecast with 80% confidence band; LightGBM vs Prophet toggle |
| 🎛️ What-If | Promo discount slider — model re-runs with modified features, shows delta vs baseline |
| 📐 Evaluation | Backtest on held-out 12 weeks — MAPE, WAPE, RMSE, 80% CI coverage |

---

## Model Architecture

**Primary: LightGBM Quantile Regression**
- Three models trained simultaneously: p10, p50, p90
- Features: lag-7/14/28, rolling mean/std, day-of-week, week-of-year, is_weekend, is_promo, sell_price, price momentum
- Quantile regression gives honest uncertainty — the p10–p90 band is learned from data, not parametric

**Comparison: Prophet with promo regressor**
- `seasonality_mode="multiplicative"` for retail data
- `is_promo` added as an additional regressor
- Native 80% uncertainty intervals

**Why both?** LightGBM handles complex feature interactions and recent patterns better. Prophet handles long-run seasonality more robustly. Showing both lets a client see where they agree and where they diverge — which is itself a signal about forecast uncertainty.

---

## What-If Scenario Logic

The promo slider doesn't apply a fixed multiplier to the forecast. It modifies the input features fed to the trained LightGBM model — specifically `sell_price` and `is_promo` — and re-runs the forward pass. The model uses its learned relationship between price reductions and sales lift from historical data. This means the response curve is data-driven, not assumed.

---

## Evaluation Metrics

| Metric | Definition |
|---|---|
| MAPE | Mean Absolute Percentage Error — intuitive but sensitive to low-volume items |
| WAPE | Weighted APE — sum(abs errors) / sum(actuals). More robust for retail |
| RMSE | Root Mean Squared Error — penalises large misses more heavily |
| 80% CI Coverage | % of actuals falling within p10–p90 band. Well-calibrated model ≈ 80% |

---

## Dataset

[M5 Forecasting — Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) (Kaggle)

Walmart daily sales data across 10 stores and 3,049 products, with calendar events and sell prices. One of the most widely used public benchmarks for retail demand forecasting.

---

## Setup

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Download M5 data from Kaggle into data/
#    Files needed: sales_train_evaluation.csv, calendar.csv, sell_prices.csv

# 3. Build sample parquet (runs once)
python scripts/build_sample.py

# 4. Run
streamlit run streamlit_app.py
```

---

## Project Structure

```
sales_forecaster/
├── streamlit_app.py          # UI — 4-tab Streamlit app
├── requirements.txt
├── scripts/
│   └── build_sample.py       # M5 raw CSV → data/m5_sample.parquet
├── forecasting/
│   ├── __init__.py
│   ├── preprocessing.py      # Feature engineering, train/val split
│   ├── modeling.py           # LightGBM quantile + Prophet, backtest
│   └── plotting.py           # Plotly charts
└── data/
    └── m5_sample.parquet     # Generated — not committed to git
```

---

## Background

Built as a portfolio demonstration of production-level forecasting thinking. The modular package structure, proper train/validation split, quantile regression for uncertainty, and what-if scenario layer reflect how forecasting systems are built in production retail environments — not how they're typically demoed.
