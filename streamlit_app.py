"""
streamlit_app.py
----------------
Multi-Horizon Sales Forecaster with Confidence Intervals and What-If Scenarios
Built on M5 Walmart dataset (public).

Tabs:
  1. Data Explorer  — history + promo periods
  2. Forecast       — multi-horizon with confidence band, model comparison toggle
  3. What-If        — promo/seasonal sliders, real-time forecast delta
  4. Evaluation     — backtesting on held-out 12 weeks, MAPE/WAPE/RMSE/Coverage
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

from forecasting.preprocessing import load_sample_data, train_val_split, make_features
from forecasting.modeling import (
    train_lightgbm,
    forecast_lightgbm,
    train_and_forecast_prophet,
    backtest,
)
from forecasting.plotting import (
    plot_history,
    plot_forecast,
    plot_whatif_comparison,
    plot_backtest,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Sales Forecaster 📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Multi-Horizon Sales Forecaster")
st.caption(
    "Demand forecasting with uncertainty quantification and what-if scenario analysis. "
    "Built on the M5 Walmart public dataset."
)

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Controls")

    # Data source
    parquet_path = "data/m5_sample.parquet"
    if not os.path.exists(parquet_path):
        st.error(
            "Sample data not found. Run `python scripts/build_sample.py` "
            "to generate data/m5_sample.parquet from the M5 raw CSVs."
        )
        st.stop()

    df_all = pd.read_parquet(parquet_path)
    df_all["date"] = pd.to_datetime(df_all["date"])
    item_ids = sorted(df_all["item_id"].unique())

    item_id = st.selectbox("Select Item", item_ids)
    df = df_all[df_all["item_id"] == item_id].copy()

    st.divider()

    # Forecast horizon
    horizon_label = st.selectbox(
        "Forecast Horizon",
        ["1 Week", "4 Weeks", "12 Weeks"],
    )
    horizon_map = {"1 Week": 7, "4 Weeks": 28, "12 Weeks": 84}
    horizon_days = horizon_map[horizon_label]

    # Model comparison toggle
    show_prophet = st.toggle("Show Prophet comparison", value=False)

    st.divider()
    st.caption("📊 Data: M5 Walmart (Kaggle) · Model: LightGBM Quantile Regression")

# ---------------------------------------------------------------------------
# Cache expensive training
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_lgbm_models(item_id: str, promo_pct: float):
    df = df_all[df_all["item_id"] == item_id].copy()
    df["date"] = pd.to_datetime(df["date"])
    train, _ = train_val_split(df, val_weeks=12)
    return train_lightgbm(train, promo_pct=promo_pct)


@st.cache_data(show_spinner=False)
def get_backtest_results(item_id: str):
    df = df_all[df_all["item_id"] == item_id].copy()
    df["date"] = pd.to_datetime(df["date"])
    train, val = train_val_split(df, val_weeks=12)
    return backtest(train, val)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Explorer",
    "🔮 Forecast",
    "🎛️ What-If Scenarios",
    "📐 Evaluation",
])

# ── Tab 1: Data Explorer ────────────────────────────────────────────────────
with tab1:
    st.subheader(f"Sales History — {item_id}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Avg Daily Sales", f"{df['sales'].mean():.1f} units")
    col3.metric("Promo Days", f"{df['is_promo'].sum():,} ({df['is_promo'].mean()*100:.1f}%)")

    st.plotly_chart(plot_history(df, item_id), use_container_width=True)

    with st.expander("Raw data sample"):
        st.dataframe(df.tail(30), use_container_width=True)


# ── Tab 2: Forecast ─────────────────────────────────────────────────────────
with tab2:
    st.subheader(f"{horizon_label} Ahead Forecast")

    with st.spinner("Training LightGBM models..."):
        models = get_lgbm_models(item_id, promo_pct=0.0)
        train_df, _ = train_val_split(df, val_weeks=12)
        lgbm_result = forecast_lightgbm(models, train_df, horizon_days)

    prophet_result = None
    if show_prophet:
        with st.spinner("Running Prophet..."):
            prophet_result = train_and_forecast_prophet(train_df, horizon_days)

    fig = plot_forecast(
        train_df,
        lgbm_result,
        baseline_result=prophet_result,
        title=f"{horizon_label} Sales Forecast — {item_id}",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Forecast summary table
    forecast_summary = pd.DataFrame({
        "Date": lgbm_result.dates,
        "Low (p10)": lgbm_result.p10.round(1),
        "Forecast (p50)": lgbm_result.p50.round(1),
        "High (p90)": lgbm_result.p90.round(1),
    })
    with st.expander("📋 Forecast table"):
        st.dataframe(forecast_summary, use_container_width=True)
        csv = forecast_summary.to_csv(index=False)
        st.download_button(
            "Download forecast CSV",
            data=csv,
            file_name=f"forecast_{item_id}_{horizon_label.replace(' ','_')}.csv",
            mime="text/csv",
        )


# ── Tab 3: What-If Scenarios ────────────────────────────────────────────────
with tab3:
    st.subheader("What-If Scenario Analysis")
    st.caption(
        "Adjust promotion and pricing assumptions to see how the forecast changes. "
        "The model re-runs with modified feature inputs — not a simple scalar adjustment."
    )

    col1, col2 = st.columns(2)
    with col1:
        promo_pct = st.slider(
            "Promo Discount %",
            min_value=0, max_value=30, value=0, step=5,
            help="Simulates a price reduction of this % during the forecast period"
        ) / 100.0

        promo_week = st.selectbox(
            "Apply promo in which week?",
            ["All weeks"] + list(range(1, 53)),
            help="Limit the promo to a specific ISO week number, or apply across all forecast weeks"
        )
        promo_week_val = None if promo_week == "All weeks" else int(promo_week)

    with col2:
        st.info(
            "💡 **How this works:** The promo discount is applied as a feature "
            "to the LightGBM model — it uses the learned relationship between "
            "price reductions and sales lift from historical data, not a fixed multiplier."
        )

    if promo_pct > 0:
        with st.spinner("Running scenario..."):
            scenario_models = get_lgbm_models(item_id, promo_pct=promo_pct)
            train_df_s, _ = train_val_split(df, val_weeks=12)
            baseline_res = forecast_lightgbm(
                get_lgbm_models(item_id, promo_pct=0.0),
                train_df_s, horizon_days, promo_pct=0.0
            )
            scenario_res = forecast_lightgbm(
                scenario_models,
                train_df_s, horizon_days,
                promo_pct=promo_pct,
                promo_week=promo_week_val,
            )

        fig = plot_whatif_comparison(baseline_res, scenario_res, promo_pct)
        st.plotly_chart(fig, use_container_width=True)

        # Delta summary
        delta_units = scenario_res.p50.sum() - baseline_res.p50.sum()
        delta_pct = delta_units / (baseline_res.p50.sum() + 1e-8) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline Total (units)", f"{baseline_res.p50.sum():,.0f}")
        c2.metric("Scenario Total (units)", f"{scenario_res.p50.sum():,.0f}")
        c3.metric("Lift", f"{delta_units:+.0f} units", f"{delta_pct:+.1f}%")
    else:
        st.info("Set a promo discount above 0% to see the scenario comparison.")


# ── Tab 4: Evaluation ───────────────────────────────────────────────────────
with tab4:
    st.subheader("Backtesting — Held-Out Last 12 Weeks")
    st.caption(
        "Model trained on all data except the last 12 weeks. "
        "Forecast generated for those 12 weeks and compared against actuals. "
        "This is how you'd evaluate the model before deploying it."
    )

    with st.spinner("Running backtest..."):
        bt_df, metrics = get_backtest_results(item_id)

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAPE", f"{metrics['MAPE']:.1f}%", help="Mean Absolute Percentage Error")
    c2.metric("WAPE", f"{metrics['WAPE']:.1f}%", help="Weighted Absolute Percentage Error — more robust than MAPE")
    c3.metric("RMSE", f"{metrics['RMSE']:.1f}", help="Root Mean Squared Error (in units)")
    c4.metric(
        "80% CI Coverage",
        f"{metrics['Coverage_80pct']:.1f}%",
        help="% of actuals falling within the p10–p90 band. Well-calibrated = ~80%"
    )

    fig, _ = plot_backtest(bt_df, metrics)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Backtest detail table"):
        st.dataframe(
            bt_df.assign(
                p50_rounded=bt_df["p50"].round(1),
                actual_rounded=bt_df["actual"].round(1),
            )[["date", "actual_rounded", "p50_rounded", "p10", "p90", "abs_error"]],
            use_container_width=True,
        )
