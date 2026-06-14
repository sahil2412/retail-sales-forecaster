"""
plotting.py
-----------
All Plotly charts for the Streamlit UI.
Each function returns a go.Figure — the app just calls st.plotly_chart().
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from forecasting.modeling import ForecastResult

# Colour palette — consistent across all charts
COLOURS = {
    "actual": "#2196F3",
    "lgbm": "#4CAF50",
    "prophet": "#FF9800",
    "band": "rgba(76, 175, 80, 0.15)",
    "promo": "rgba(255, 87, 34, 0.2)",
    "error": "#F44336",
}


def plot_history(df: pd.DataFrame, item_id: str) -> go.Figure:
    """Tab 1 — raw sales history with promo periods highlighted."""
    fig = go.Figure()

    # Promo shading
    promo_dates = df[df["is_promo"] == 1]["date"]
    for d in promo_dates:
        fig.add_vrect(
            x0=d - pd.Timedelta(hours=12),
            x1=d + pd.Timedelta(hours=12),
            fillcolor=COLOURS["promo"],
            layer="below",
            line_width=0,
        )

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["sales"],
        mode="lines",
        name="Daily Sales",
        line=dict(color=COLOURS["actual"], width=1.5),
    ))

    fig.update_layout(
        title=f"Sales History — {item_id}",
        xaxis_title="Date",
        yaxis_title="Units Sold",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
        height=400,
        margin=dict(t=50, b=60),
    )
    return fig


def plot_forecast(
    history_df: pd.DataFrame,
    result: ForecastResult,
    baseline_result: ForecastResult = None,
    title: str = "Sales Forecast",
) -> go.Figure:
    """
    Tab 2 — forecast with confidence band.
    Optional baseline_result for model comparison toggle.
    """
    fig = go.Figure()

    # Historical actuals (last 60 days for context)
    hist = history_df.tail(60)
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["sales"],
        mode="lines",
        name="Historical",
        line=dict(color=COLOURS["actual"], width=1.5),
    ))

    # Primary model confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([result.dates, result.dates[::-1]]),
        y=np.concatenate([result.p90, result.p10[::-1]]),
        fill="toself",
        fillcolor=COLOURS["band"],
        line=dict(color="rgba(0,0,0,0)"),
        name="80% Confidence Interval",
        showlegend=True,
    ))

    # Primary model p50
    fig.add_trace(go.Scatter(
        x=result.dates, y=result.p50,
        mode="lines",
        name=result.model_name,
        line=dict(color=COLOURS["lgbm"], width=2),
    ))

    # Optional Prophet comparison
    if baseline_result is not None:
        fig.add_trace(go.Scatter(
            x=baseline_result.dates, y=baseline_result.p50,
            mode="lines",
            name=baseline_result.model_name,
            line=dict(color=COLOURS["prophet"], width=2, dash="dash"),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Units Sold",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
        height=450,
        margin=dict(t=50, b=70),
    )
    return fig


def plot_whatif_comparison(
    baseline: ForecastResult,
    scenario: ForecastResult,
    promo_pct: float,
) -> go.Figure:
    """Tab 3 — side-by-side baseline vs what-if scenario."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=baseline.dates, y=baseline.p50,
        mode="lines",
        name="Baseline (no promo)",
        line=dict(color=COLOURS["actual"], width=2),
    ))

    # Scenario confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([scenario.dates, scenario.dates[::-1]]),
        y=np.concatenate([scenario.p90, scenario.p10[::-1]]),
        fill="toself",
        fillcolor="rgba(255, 152, 0, 0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Scenario 80% CI",
        showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=scenario.dates, y=scenario.p50,
        mode="lines",
        name=f"Scenario ({int(promo_pct*100)}% promo)",
        line=dict(color=COLOURS["prophet"], width=2),
    ))

    # Delta annotation
    delta = scenario.p50.sum() - baseline.p50.sum()
    delta_pct = delta / (baseline.p50.sum() + 1e-8) * 100
    fig.add_annotation(
        x=scenario.dates.iloc[-1],
        y=scenario.p50.max(),
        text=f"Δ {delta:+.0f} units ({delta_pct:+.1f}%)",
        showarrow=True,
        arrowhead=2,
        bgcolor="#FF9800",
        font=dict(color="white", size=12),
    )

    fig.update_layout(
        title=f"What-If: {int(promo_pct*100)}% Promo vs Baseline",
        xaxis_title="Date",
        yaxis_title="Units Sold",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
        height=450,
        margin=dict(t=50, b=70),
    )
    return fig


def plot_backtest(backtest_df: pd.DataFrame, metrics: dict) -> go.Figure:
    """Tab 4 — actual vs predicted on held-out validation period."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Actual vs Predicted (held-out 12 weeks)", "Absolute Error by Day"],
        vertical_spacing=0.15,
        row_heights=[0.65, 0.35],
    )

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([backtest_df["date"], backtest_df["date"][::-1]]),
        y=pd.concat([backtest_df["p90"], backtest_df["p10"][::-1]]),
        fill="toself",
        fillcolor=COLOURS["band"],
        line=dict(color="rgba(0,0,0,0)"),
        name="80% CI",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=backtest_df["date"], y=backtest_df["actual"],
        mode="lines",
        name="Actual",
        line=dict(color=COLOURS["actual"], width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=backtest_df["date"], y=backtest_df["p50"],
        mode="lines",
        name="Predicted (p50)",
        line=dict(color=COLOURS["lgbm"], width=2),
    ), row=1, col=1)

    # Error bars
    fig.add_trace(go.Bar(
        x=backtest_df["date"],
        y=backtest_df["abs_error"],
        name="Abs Error",
        marker_color=COLOURS["error"],
        opacity=0.6,
    ), row=2, col=1)

    fig.update_layout(
        height=650,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.08),
        margin=dict(t=60, b=40),
    )
    return fig, metrics
