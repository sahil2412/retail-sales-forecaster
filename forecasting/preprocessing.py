"""
preprocessing.py
----------------
Data loading, cleaning, and feature engineering for the M5 sales forecaster.
Handles the M5 Walmart dataset structure (sales_train_evaluation.csv + calendar.csv + sell_prices.csv).
Also supports a simplified single-store CSV for demo/upload mode.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# M5 sample builder — called once to create data/m5_sample.parquet
# ---------------------------------------------------------------------------

def build_m5_sample(
    sales_path: str,
    calendar_path: str,
    prices_path: str,
    n_items: int = 10,
    store_id: str = "CA_1",
) -> pd.DataFrame:
    """
    Reads raw M5 CSVs and returns a long-format DataFrame with columns:
        item_id, date, sales, sell_price, event_name, is_promo
    Saves to data/m5_sample.parquet for fast reloads.
    """
    sales = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path, parse_dates=["date"])
    prices = pd.read_csv(prices_path)

    # Filter to one store, top N items by total sales
    store_cols = [c for c in sales.columns if c.startswith("d_")]
    store_sales = sales[sales["store_id"] == store_id].copy()
    store_sales["total"] = store_sales[store_cols].sum(axis=1)
    top_items = store_sales.nlargest(n_items, "total")["item_id"].tolist()
    store_sales = store_sales[store_sales["item_id"].isin(top_items)]

    # Melt to long format
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    long = store_sales.melt(id_vars=id_cols, value_vars=store_cols,
                             var_name="d", value_name="sales")

    # Merge calendar
    long = long.merge(calendar[["d", "date", "event_name_1", "snap_CA"]], on="d", how="left")
    long = long.rename(columns={"event_name_1": "event_name"})

    # Merge prices
    prices_store = prices[prices["store_id"] == store_id][["item_id", "wm_yr_wk", "sell_price"]]
    cal_wk = calendar[["d", "wm_yr_wk"]]
    long = long.merge(cal_wk, on="d", how="left")
    long = long.merge(prices_store, on=["item_id", "wm_yr_wk"], how="left")

    # Promo flag: price drop > 5% vs rolling 4-week average
    long = long.sort_values(["item_id", "date"])
    long["price_roll4"] = (
        long.groupby("item_id")["sell_price"]
        .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    )
    long["is_promo"] = (
        (long["sell_price"] < long["price_roll4"] * 0.95)
        .fillna(False)
        .astype(int)
    )

    long = long[["item_id", "date", "sales", "sell_price", "event_name", "is_promo", "snap_CA"]]
    long.to_parquet("data/m5_sample.parquet", index=False)
    return long


# ---------------------------------------------------------------------------
# Feature engineering — called by modeling.py
# ---------------------------------------------------------------------------

def make_features(df: pd.DataFrame, promo_pct: float = 0.0) -> pd.DataFrame:
    """
    Adds lag features, rolling statistics, calendar features, and promo flag.
    promo_pct: what-if promo discount percentage (0.0 = no promo, 0.2 = 20% off)
    """
    df = df.copy().sort_values("date")

    # Override promo flag for what-if scenarios
    if promo_pct > 0:
        df["is_promo"] = 1
        df["sell_price"] = df["sell_price"] * (1 - promo_pct)

    # Calendar features
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Event flag
    df["has_event"] = df["event_name"].notna().astype(int)

    # Lag features
    for lag in [7, 14, 28]:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    # Rolling statistics (on past data only — no leakage)
    for window in [7, 28]:
        df[f"rolling_mean_{window}"] = df["sales"].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"] = df["sales"].shift(1).rolling(window).std()

    # Price momentum
    df["price_change_1w"] = df["sell_price"].pct_change(7).fillna(0)

    df = df.dropna()
    return df


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def train_val_split(df: pd.DataFrame, val_weeks: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits on time. Last val_weeks * 7 days become the validation set.
    Used for backtesting in the evaluation tab.
    """
    cutoff = df["date"].max() - pd.Timedelta(weeks=val_weeks)
    train = df[df["date"] <= cutoff].copy()
    val = df[df["date"] > cutoff].copy()
    return train, val


# ---------------------------------------------------------------------------
# Synthetic fallback — used when m5_sample.parquet doesn't exist (Streamlit Cloud)
# ---------------------------------------------------------------------------

SYNTHETIC_ITEMS = [
    "FOODS_1_001", "FOODS_1_002", "FOODS_2_001",
    "HOBBIES_1_001", "HOBBIES_1_002",
    "HOUSEHOLD_1_001", "HOUSEHOLD_1_002",
    "FOODS_3_001", "FOODS_3_002", "FOODS_3_003",
]


def build_synthetic_sample(n_days: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic-looking retail sales data when M5 CSVs are not available.
    Mimics M5 structure: daily sales with seasonality, trend, price, promo, events.
    Saves to data/m5_sample.parquet so subsequent loads are fast.
    """
    rng = np.random.RandomState(seed)
    os.makedirs("data", exist_ok=True)
    rows = []
    base_date = pd.Timestamp("2013-01-01")
    dates = pd.date_range(base_date, periods=n_days, freq="D")

    for item_id in SYNTHETIC_ITEMS:
        # Per-item baseline and noise characteristics
        base_sales = rng.randint(30, 120)
        price = round(rng.uniform(1.5, 9.9), 2)
        trend = rng.uniform(-0.005, 0.01)

        for i, date in enumerate(dates):
            # Seasonality: weekly + annual
            weekly = 1.3 if date.dayofweek in [5, 6] else 1.0
            annual = 1 + 0.2 * np.sin(2 * np.pi * date.dayofyear / 365)
            trend_mult = 1 + trend * i

            # Promo: random ~8% of days
            is_promo = int(rng.random() < 0.08)
            promo_lift = 1.25 if is_promo else 1.0
            promo_price = round(price * 0.85, 2) if is_promo else price

            # Event: ~5% of days
            event = "Holiday" if rng.random() < 0.05 else None
            event_lift = 1.15 if event else 1.0

            # Final sales with noise
            mu = max(1.0, abs(base_sales * weekly * annual * trend_mult * promo_lift * event_lift))
            sales = int(rng.poisson(mu))

            rows.append({
                "item_id": item_id,
                "date": date,
                "sales": sales,
                "sell_price": promo_price,
                "event_name": event,
                "is_promo": is_promo,
                "snap_CA": int(rng.random() < 0.3),
            })

    df = pd.DataFrame(rows)
    df.to_parquet("data/m5_sample.parquet", index=False)
    return df


# ---------------------------------------------------------------------------
# Load helper for Streamlit — with synthetic fallback
# ---------------------------------------------------------------------------

def load_sample_data(item_id: str, parquet_path: str = "data/m5_sample.parquet") -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    return df[df["item_id"] == item_id].copy()


def load_or_build_data(parquet_path: str = "data/m5_sample.parquet") -> pd.DataFrame:
    """
    Loads parquet if it exists, otherwise builds synthetic data.
    Called by streamlit_app.py — handles both local (M5) and deployed (synthetic) environments.
    """
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    # Streamlit Cloud deploy — no M5 CSVs available
    return build_synthetic_sample()
