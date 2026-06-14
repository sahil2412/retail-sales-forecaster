"""
modeling.py
-----------
Two parallel models:
  1. LightGBM quantile regression (p10, p50, p90) — primary model
  2. Prophet with promo regressor — comparison baseline

Both return a unified ForecastResult dataclass so the UI layer
doesn't care which model produced the numbers.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb
from prophet import Prophet
from sklearn.metrics import mean_squared_error

from forecasting.preprocessing import make_features


# ---------------------------------------------------------------------------
# Shared result container
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    dates: pd.Series
    p10: np.ndarray       # lower bound
    p50: np.ndarray       # median / point forecast
    p90: np.ndarray       # upper bound
    model_name: str


# ---------------------------------------------------------------------------
# Feature columns used by LightGBM
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "dayofweek", "month", "weekofyear", "is_weekend",
    "has_event", "is_promo", "sell_price", "price_change_1w",
    "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_28",
    "rolling_std_7", "rolling_std_28",
]


# ---------------------------------------------------------------------------
# LightGBM quantile model
# ---------------------------------------------------------------------------

def _lgb_params(quantile: float) -> dict:
    return {
        "objective": "quantile",
        "alpha": quantile,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": -1,
    }


def train_lightgbm(
    train_df: pd.DataFrame,
    promo_pct: float = 0.0,
) -> dict:
    """
    Trains three LightGBM models (p10, p50, p90) on train_df.
    Returns dict of fitted models keyed by quantile.
    """
    df = make_features(train_df, promo_pct=promo_pct)
    X = df[FEATURE_COLS]
    y = df["sales"]

    models = {}
    for q in [0.1, 0.5, 0.9]:
        m = lgb.LGBMRegressor(**_lgb_params(q))
        m.fit(X, y)
        models[q] = m
    return models


def forecast_lightgbm(
    models: dict,
    history_df: pd.DataFrame,
    horizon_days: int,
    promo_pct: float = 0.0,
    promo_week: Optional[int] = None,
) -> ForecastResult:
    """
    Generates horizon_days ahead forecast using recursive prediction.
    promo_pct: discount fraction applied during promo_week (what-if).
    promo_week: ISO week number to apply the promo (None = apply throughout).
    """
    df = history_df.copy().sort_values("date")

    future_rows = []
    last_price = df["sell_price"].iloc[-1]
    last_event = None

    for i in range(horizon_days):
        next_date = df["date"].iloc[-1] + pd.Timedelta(days=1)
        week_num = next_date.isocalendar()[1]

        # What-if promo logic
        active_promo = promo_pct > 0 and (promo_week is None or week_num == promo_week)
        price = last_price * (1 - promo_pct) if active_promo else last_price

        new_row = {
            "date": next_date,
            "sales": np.nan,
            "sell_price": price,
            "event_name": last_event,
            "is_promo": int(active_promo),
            "snap_CA": 0,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        featurised = make_features(df, promo_pct=0)  # already applied above
        if len(featurised) == 0:
            continue

        last_features = featurised.iloc[[-1]][FEATURE_COLS]

        preds = {q: max(0, models[q].predict(last_features)[0]) for q in [0.1, 0.5, 0.9]}

        # Fill the sales column with p50 for next step's lag calculation
        df.loc[df.index[-1], "sales"] = preds[0.5]
        future_rows.append({
            "date": next_date,
            "p10": preds[0.1],
            "p50": preds[0.5],
            "p90": preds[0.9],
        })

    result_df = pd.DataFrame(future_rows)
    return ForecastResult(
        dates=result_df["date"],
        p10=result_df["p10"].values,
        p50=result_df["p50"].values,
        p90=result_df["p90"].values,
        model_name="LightGBM (Quantile)",
    )


# ---------------------------------------------------------------------------
# Prophet model
# ---------------------------------------------------------------------------

def train_and_forecast_prophet(
    train_df: pd.DataFrame,
    horizon_days: int,
    promo_pct: float = 0.0,
) -> ForecastResult:
    """
    Fits Prophet with is_promo as an additional regressor.
    Returns ForecastResult with Prophet's native uncertainty intervals.
    """
    df = train_df[["date", "sales", "is_promo"]].copy()
    df = df.rename(columns={"date": "ds", "sales": "y"})

    if promo_pct > 0:
        df["is_promo"] = 1

    m = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.8,
    )
    m.add_regressor("is_promo")
    m.fit(df)

    future = m.make_future_dataframe(periods=horizon_days)
    future["is_promo"] = int(promo_pct > 0)
    forecast = m.predict(future)
    forecast = forecast.tail(horizon_days)

    return ForecastResult(
        dates=forecast["ds"],
        p10=np.maximum(0, forecast["yhat_lower"].values),
        p50=np.maximum(0, forecast["yhat"].values),
        p90=np.maximum(0, forecast["yhat_upper"].values),
        model_name="Prophet",
    )


# ---------------------------------------------------------------------------
# Backtesting — used by evaluation tab
# ---------------------------------------------------------------------------

def backtest(
    df: pd.DataFrame,
    val_df: pd.DataFrame,
    promo_pct: float = 0.0,
) -> pd.DataFrame:
    """
    Trains LightGBM on df (train), forecasts len(val_df) days,
    compares against val_df actuals.
    Returns DataFrame with date, actual, p10, p50, p90, mape, wape columns.
    """
    models = train_lightgbm(df, promo_pct=promo_pct)
    result = forecast_lightgbm(models, df, horizon_days=len(val_df), promo_pct=promo_pct)

    actuals = val_df["sales"].values[:len(result.p50)]
    dates = val_df["date"].values[:len(result.p50)]

    # Per-horizon metrics
    mape = np.abs((actuals - result.p50) / (actuals + 1e-8)) * 100
    wape = np.sum(np.abs(actuals - result.p50)) / (np.sum(actuals) + 1e-8) * 100
    rmse = np.sqrt(mean_squared_error(actuals, result.p50))

    return pd.DataFrame({
        "date": dates,
        "actual": actuals,
        "p10": result.p10,
        "p50": result.p50,
        "p90": result.p90,
        "abs_error": np.abs(actuals - result.p50),
    }), {
        "MAPE": round(float(np.mean(mape)), 2),
        "WAPE": round(float(wape), 2),
        "RMSE": round(float(rmse), 2),
        "Coverage_80pct": round(
            float(np.mean((actuals >= result.p10) & (actuals <= result.p90))) * 100, 1
        ),
    }
