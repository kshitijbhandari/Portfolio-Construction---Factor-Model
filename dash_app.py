"""
Fama-French Factor Model Portfolio Optimizer – Dash version
Runs alongside the Streamlit app (app.py); does NOT modify it.
Deploy on Render.com:  gunicorn dash_app:server --timeout 600
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Data loading ──────────────────────────────────────────────────────────────
DATA_DIR = "data"

def _load_data():
    sr   = pd.read_csv(os.path.join(DATA_DIR, "nifty_stocks_data (1).csv"))
    idx  = pd.read_csv(os.path.join(DATA_DIR, "nifty50_index_data.csv"))
    ff   = pd.read_csv(os.path.join(DATA_DIR, "FF_Nifty50.csv"))
    univ = pd.read_csv(os.path.join(DATA_DIR, "Nifty_50.csv"))
    return sr, idx, ff, univ

stock_returns_data, index_returns_data, fama_french_data, yearly_tickers_data = _load_data()
stock_returns_data["Date"] = pd.to_datetime(stock_returns_data["Date"])

all_months   = pd.period_range(
    start=stock_returns_data["Date"].min(),
    end=stock_returns_data["Date"].max(),
    freq="M"
)
oos_options   = [{"label": str(m), "value": str(m)} for m in all_months]
default_oos   = oos_options[max(0, len(oos_options) - 60)]["value"]

# ── App init ──────────────────────────────────────────────────────────────────
app    = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="Factor Model Portfolio Optimizer",
)
server = app.server   # gunicorn entry-point

# ── Sidebar ───────────────────────────────────────────────────────────────────
_slider_tip = {"placement": "bottom", "always_visible": False}

sidebar = dbc.Col(
    [
        html.H4("⚙️ Configuration", className="text-primary mb-3"),

        # ── Backtest params ─────────────────────────────────────────────────
        html.H6("📈 Backtest Parameters", className="text-secondary mt-2"),
        dbc.Label("Out-of-Sample Start", html_for="oos-start"),
        dcc.Dropdown(
            id="oos-start",
            options=oos_options,
            value=default_oos,
            clearable=False,
            style={"color": "#000"},
        ),
        html.Br(),
        dbc.Label("OOS Duration (months)"),
        dcc.Slider(id="oos-months",  min=6,  max=48,  step=1,  value=24,
                   marks={6:"6", 24:"24", 48:"48"}, tooltip=_slider_tip),
        dbc.Label("Lookback Period (months)"),
        dcc.Slider(id="lookback-months", min=12, max=120, step=12, value=36,
                   marks={12:"12", 60:"60", 120:"120"}, tooltip=_slider_tip),
        dbc.Label("Rebalance Frequency (months)"),
        dcc.Slider(id="rebalance-every", min=1, max=12, step=1, value=3,
                   marks={1:"1", 3:"3", 6:"6", 12:"12"}, tooltip=_slider_tip),

        html.Hr(className="border-secondary"),

        # ── Optimization ────────────────────────────────────────────────────
        html.H6("🎯 Optimization Constraints", className="text-secondary"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Max Positions"),
                dcc.Slider(id="k-max", min=5, max=50, step=1, value=15,
                           marks={5:"5", 15:"15", 50:"50"}, tooltip=_slider_tip),
            ], width=6),
            dbc.Col([
                dbc.Label("Max Position Size"),
                dcc.Slider(id="w-max", min=0.05, max=0.5, step=0.05, value=0.20,
                           marks={0.05:"5%", 0.20:"20%", 0.5:"50%"}, tooltip=_slider_tip),
            ], width=6),
        ]),
        dbc.Label("Risk Aversion"),
        dcc.Slider(id="risk-aversion", min=0.1, max=10.0, step=0.1, value=1.0,
                   marks={0.1:"0.1", 5:"5", 10:"10"}, tooltip=_slider_tip),

        html.Hr(className="border-secondary"),

        # ── Betas ───────────────────────────────────────────────────────────
        html.H6("📊 Target Betas & Tolerances", className="text-secondary"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Target Betas"),
                dbc.InputGroup([dbc.InputGroupText("MF"),  dbc.Input(id="target-mf",  type="number", value=1.0,  step=0.1)], className="mb-1"),
                dbc.InputGroup([dbc.InputGroupText("SMB"), dbc.Input(id="target-smb", type="number", value=0.0,  step=0.1)], className="mb-1"),
                dbc.InputGroup([dbc.InputGroupText("HML"), dbc.Input(id="target-hml", type="number", value=0.2,  step=0.1)], className="mb-1"),
            ], width=6),
            dbc.Col([
                dbc.Label("Tolerances (±)"),
                dbc.InputGroup([dbc.InputGroupText("MF"),  dbc.Input(id="tol-mf",  type="number", value=0.3, step=0.05)], className="mb-1"),
                dbc.InputGroup([dbc.InputGroupText("SMB"), dbc.Input(id="tol-smb", type="number", value=0.3, step=0.05)], className="mb-1"),
                dbc.InputGroup([dbc.InputGroupText("HML"), dbc.Input(id="tol-hml", type="number", value=0.3, step=0.05)], className="mb-1"),
            ], width=6),
        ]),

        html.Hr(className="border-secondary"),

        # ── Turnover cap ────────────────────────────────────────────────────
        html.H6("🔁 Turnover Constraint", className="text-secondary"),
        dbc.Checklist(
            id="use-turnover-cap",
            options=[{"label": " Enable Turnover Cap", "value": "on"}],
            value=[],
            switch=True,
        ),
        html.Div(
            id="turnover-slider-wrap",
            children=[
                dbc.Label("Turnover Cap (fraction)"),
                dcc.Slider(id="turnover-cap", min=0.05, max=1.0, step=0.05, value=0.30,
                           marks={0.05:"5%", 0.30:"30%", 1.0:"100%"}, tooltip=_slider_tip),
            ],
            style={"display": "none"},
        ),

        html.Hr(className="border-secondary"),

        # ── Capital ─────────────────────────────────────────────────────────
        dbc.Label("Initial Capital ($)"),
        dbc.Input(id="initial-capital", type="number",
                  value=100000, min=10000, max=10000000, step=10000),

        html.Br(),
    ],
    width=3,
    className="bg-dark border-end border-secondary p-3",
    style={"minHeight": "100vh", "overflowY": "auto", "position": "sticky", "top": 0},
)

# ── Tab content (all pre-rendered so callbacks resolve IDs at startup) ─────────

# ··· Beta Explorer ···
tab_beta = html.Div([
    html.H4("Beta Explorer"),
    html.P("Understand the factor landscape before committing to target betas.", className="text-muted"),

    html.H5("📈 Factor Monthly Returns over Lookback Window"),
    html.Small("Each panel shows actual monthly factor values over the selected lookback period.", className="text-muted"),
    dcc.Loading(dcc.Graph(id="factor-trend-chart"), type="circle", color="#0d6efd"),

    html.Hr(className="border-secondary"),

    html.H5("🎯 Achievable Beta Ranges"),
    dbc.Alert(
        "Computes min/max reachable portfolio beta via MILP. "
        "Shows whether your target betas are feasible before running the backtest.",
        color="info", className="py-2",
    ),
    dbc.Button("⚡ Compute Beta Ranges", id="compute-beta-btn", color="primary", className="mb-3"),
    dcc.Loading(html.Div(id="beta-ranges-output"), type="dot", color="#0d6efd"),
], className="p-3")

# ··· Run Backtest ···
tab_run = html.Div([
    html.H4("Run Portfolio Backtest"),
    html.Div(id="backtest-config-summary"),
    dbc.Button("▶️ Run Backtest", id="run-backtest-btn",
               color="primary", size="lg", className="mb-3"),
    dcc.Loading(html.Div(id="backtest-run-output"), type="default", color="#0d6efd"),
], className="p-3")

# ··· Results ···
tab_results = html.Div([
    html.H4("Backtest Results"),
    dcc.Loading(html.Div(id="results-content"), type="default", color="#0d6efd"),
], className="p-3")

# ··· Risk Analysis ···
tab_risk = html.Div([
    html.H4("Risk Sensitivity Analysis"),
    dbc.Alert("Run multiple backtests with varying risk aversion parameters.", color="secondary"),
    dbc.Row([
        dbc.Col([
            dbc.Label("Min Risk Aversion"),
            dcc.Slider(id="min-ra", min=0.1, max=5.0, step=0.1, value=1.0,
                       marks={0.1:"0.1", 2.5:"2.5", 5:"5"}, tooltip=_slider_tip),
        ], width=6),
        dbc.Col([
            dbc.Label("Max Risk Aversion"),
            dcc.Slider(id="max-ra", min=1.0, max=10.0, step=0.1, value=5.0,
                       marks={1:"1", 5:"5", 10:"10"}, tooltip=_slider_tip),
        ], width=6),
    ]),
    dbc.Label("Number of Scenarios", className="mt-2"),
    dcc.Slider(id="num-scenarios", min=2, max=20, step=1, value=5,
               marks={2:"2", 10:"10", 20:"20"}, tooltip=_slider_tip),
    html.Br(),
    dbc.Button("🔄 Run Risk Sensitivity Analysis", id="run-sensitivity-btn",
               color="warning", className="mb-3"),
    dcc.Loading(html.Div(id="sensitivity-output"), type="default", color="#ffc107"),
], className="p-3")

# ··· Info ···
tab_info = html.Div([
    dcc.Markdown("""
## 📚 Fama-French 3-Factor Model

This application implements a portfolio optimization framework based on the
**Fama-French 3-Factor Model**:

1. **Market Factor (MF)**: Broad market risk exposure
2. **Size Factor (SMB)**: Small-cap vs large-cap premium
3. **Value Factor (HML)**: High book-to-market vs low premium

## 🎯 Optimization Approach

- **Objective**: Maximize expected return while penalizing risk (Mean Absolute Deviation)
- **Constraints**:
  - Fully invested portfolio
  - Target factor exposures with user-defined tolerances
  - Cardinality constraint (max positions)
  - Position size limits

## 📊 Key Features

- **Rolling Beta Estimation**: 36-60 month rolling window OLS regression
- **Quarterly Rebalancing**: Breach-triggered rebalancing when factor exposures drift
- **MILP Optimization**: Mixed-Integer Linear Programming via PuLP/CBC solver
- **Sensitivity Analysis**: Test multiple risk aversion parameters

## 📁 Data Structure

Required CSV files in the `data/` directory:
- `nifty_stocks_data (1).csv` – monthly stock returns
- `nifty50_index_data.csv` – benchmark index returns
- `FF_Nifty50.csv` – Fama-French factors (MF, SMB, HML, RF)
- `Nifty_50.csv` – available tickers by year

## ⚠️ Interpretation Guide

**When optimization is infeasible:**
- Use the Beta Explorer tab to see achievable ranges first
- Adjust target betas to fall within those ranges
- Increase tolerances if constraints are too tight
"""),
], className="p-3")

# ── Full layout ────────────────────────────────────────────────────────────────
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                sidebar,
                dbc.Col(
                    [
                        html.H2("📊 Fama-French Factor Model Portfolio Optimizer",
                                className="text-primary mb-2 mt-3"),
                        html.Hr(className="border-secondary"),
                        dbc.Tabs(
                            [
                                dbc.Tab(tab_beta,    label="📐 Beta Explorer", tab_id="tab-beta"),
                                dbc.Tab(tab_run,     label="📊 Run Backtest",  tab_id="tab-run"),
                                dbc.Tab(tab_results, label="📈 Results",       tab_id="tab-results"),
                                dbc.Tab(tab_risk,    label="🔍 Risk Analysis", tab_id="tab-risk"),
                                dbc.Tab(tab_info,    label="ℹ️ Info",          tab_id="tab-info"),
                            ],
                            active_tab="tab-beta",
                        ),
                        # Persistent data store
                        dcc.Store(id="backtest-store"),
                    ],
                    width=9,
                    className="p-3",
                ),
            ],
            className="g-0",
        ),
        html.Footer(
            html.Small("Factor Model Portfolio Optimizer | Dash + Render.com",
                       className="text-muted"),
            className="text-center py-3 border-top border-secondary",
        ),
    ],
    fluid=True,
    className="bg-dark text-light",
    style={"minHeight": "100vh"},
)

# ═══════════════════════════════ CALLBACKS ════════════════════════════════════

# ── Show/hide turnover slider ──────────────────────────────────────────────────
@app.callback(
    Output("turnover-slider-wrap", "style"),
    Input("use-turnover-cap", "value"),
)
def toggle_turnover(val):
    return {"display": "block"} if val and "on" in val else {"display": "none"}


# ── Factor trend chart (Beta Explorer, auto-updates with sidebar params) ───────
@app.callback(
    Output("factor-trend-chart", "figure"),
    Input("oos-start", "value"),
    Input("lookback-months", "value"),
)
def update_factor_trend(oos_start, lookback_months):
    lookback_months = lookback_months or 36
    try:
        ff = fama_french_data.copy()
        ff["Date"] = pd.to_datetime(ff["Date"])
        ff = ff.sort_values("Date").set_index("Date")

        factor_cols  = [c for c in ["MF", "SMB", "HML"] if c in ff.columns]
        oos_dt       = pd.Period(oos_start, "M").to_timestamp(how="end")
        lb_dt        = oos_dt - pd.DateOffset(months=lookback_months)
        ff_win       = ff.loc[lb_dt:oos_dt, factor_cols].copy()
        month_labels = [str(p) for p in ff_win.index.to_period("M")]

        colors_map  = {"MF": "#1f77b4", "SMB": "#2ca02c", "HML": "#d62728"}
        factor_full = {"MF": "Market (MF)", "SMB": "Size (SMB)", "HML": "Value (HML)"}

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=[factor_full.get(c, c) for c in factor_cols],
            vertical_spacing=0.07,
        )

        for i, col in enumerate(factor_cols, 1):
            series     = ff_win[col] * 100
            base_color = colors_map[col]
            bar_colors = [base_color if v >= 0 else "#cc2200" for v in series.values]

            fig.add_trace(go.Bar(
                x=month_labels, y=series.values,
                name=col, marker_color=bar_colors, opacity=0.75,
                showlegend=False,
            ), row=i, col=1)

            roll3 = series.rolling(3).mean()
            fig.add_trace(go.Scatter(
                x=month_labels, y=roll3.values,
                name="3m avg",
                line=dict(color="white", dash="dash", width=1.5),
                showlegend=(i == 1),
            ), row=i, col=1)

            fig.update_yaxes(title_text=f"{col} (%)", row=i, col=1)

        fig.update_xaxes(title_text="Month", row=3, col=1, tickangle=-45)
        fig.update_layout(
            height=560,
            title_text=(f"Factor Monthly Returns "
                        f"({lb_dt.strftime('%Y-%m')} → {oos_dt.strftime('%Y-%m')})"),
            template="plotly_dark",
            legend=dict(orientation="h", y=1.03, x=0),
            margin=dict(t=90, b=60),
        )
        return fig

    except Exception as exc:
        fig = go.Figure()
        fig.add_annotation(text=f"Error rendering chart: {exc}",
                           x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(color="red"))
        fig.update_layout(template="plotly_dark")
        return fig


# ── Achievable beta ranges (MILP, triggered by button) ────────────────────────
@app.callback(
    Output("beta-ranges-output", "children"),
    Input("compute-beta-btn", "n_clicks"),
    State("oos-start", "value"),
    State("lookback-months", "value"),
    State("k-max", "value"),
    State("w-max", "value"),
    State("target-mf", "value"),
    State("target-smb", "value"),
    State("target-hml", "value"),
    prevent_initial_call=True,
)
def compute_beta_ranges(_, oos_start, lookback_months, k_max, w_max,
                        target_mf, target_smb, target_hml):
    from utils import estimate_betas_asof_nifty, compute_achievable_beta_bounds

    lookback_months = int(lookback_months or 36)
    k_max           = int(k_max or 15)
    w_max           = float(w_max or 0.20)
    target_betas    = {
        "MF":  float(target_mf  or 1.0),
        "SMB": float(target_smb or 0.0),
        "HML": float(target_hml or 0.2),
    }

    try:
        year_sel    = pd.Period(oos_start, "M").to_timestamp().year
        tickers_sel = (
            yearly_tickers_data[yearly_tickers_data["Year"] == year_sel]
            .drop(columns=["Year"]).iloc[0].dropna().tolist()
        )

        betas_sel = estimate_betas_asof_nifty(
            returns_df=stock_returns_data,
            factors_df=fama_french_data,
            asof=oos_start,
            tickers_in_window=tickers_sel,
            lookback_months=lookback_months,
            min_obs=24,
            use_t_as_last_obs=False,
        )

        sr      = stock_returns_data.copy()
        asof_dt = pd.Period(oos_start, "M").to_timestamp(how="end")
        lb_dt   = asof_dt - pd.DateOffset(months=lookback_months)
        sr_win  = sr[(sr["Date"] > lb_dt) & (sr["Date"] <= asof_dt)]
        R_sel   = sr_win.pivot(index="Date", columns="Ticker", values="RET")
        R_sel   = R_sel[R_sel.columns.intersection(tickers_sel)].dropna(axis=1, thresh=24)

        bounds = compute_achievable_beta_bounds(
            R=R_sel, betas_asof=betas_sel, K_max=k_max, w_max=w_max,
        )

        if "error" in bounds:
            return dbc.Alert(f"Cannot compute ranges: {bounds['error']}", color="danger")

        # ── Table ────────────────────────────────────────────────────────────
        rows = []
        for f, v in bounds.items():
            tgt      = target_betas.get(f, 0)
            in_range = "✅" if (v["min"] - 0.01 <= tgt <= v["max"] + 0.01) else "❌"
            rows.append({
                "Factor": f,
                "Min achievable": round(v["min"], 3),
                "Max achievable": round(v["max"], 3),
                "Your target":   round(tgt, 3),
                "In range?":     in_range,
            })

        table = dbc.Table.from_dataframe(
            pd.DataFrame(rows), striped=True, bordered=True, hover=True,
            color="dark", class_name="mb-3",
        )

        # ── Bar chart ─────────────────────────────────────────────────────────
        colors_b = {"MF": "#1f77b4", "SMB": "#2ca02c", "HML": "#d62728"}
        fig      = go.Figure()

        for xi, (f, v) in enumerate(bounds.items()):
            tgt = target_betas.get(f, 0)
            fig.add_trace(go.Bar(
                x=[f], y=[v["max"] - v["min"]], base=[v["min"]],
                name=f"Achievable range ({f})",
                marker_color=colors_b.get(f, "#888"),
                opacity=0.45, width=0.45,
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=[f], y=[tgt],
                mode="markers+text",
                marker=dict(symbol="diamond", size=14, color="white"),
                text=[f"↑ {tgt:.2f}"],
                textposition="top center",
                name="Your target" if xi == 0 else None,
                showlegend=(xi == 0),
            ))

        fig.update_layout(
            template="plotly_dark",
            barmode="overlay",
            title="Achievable Beta Ranges vs Your Targets",
            yaxis_title="Beta",
            height=370,
            margin=dict(t=55),
        )

        return html.Div([table, dcc.Graph(figure=fig)])

    except Exception as exc:
        return dbc.Alert(f"Error computing beta ranges: {exc}", color="danger")


# ── Backtest config summary ────────────────────────────────────────────────────
@app.callback(
    Output("backtest-config-summary", "children"),
    Input("oos-start", "value"),
    Input("oos-months", "value"),
    Input("lookback-months", "value"),
    Input("rebalance-every", "value"),
    Input("k-max", "value"),
    Input("w-max", "value"),
    Input("target-mf", "value"),
    Input("target-smb", "value"),
    Input("target-hml", "value"),
    Input("tol-mf", "value"),
    Input("tol-smb", "value"),
    Input("tol-hml", "value"),
    Input("use-turnover-cap", "value"),
    Input("turnover-cap", "value"),
)
def update_config_summary(oos_start, oos_months, lookback, rebalance_every,
                          k_max, w_max, target_mf, target_smb, target_hml,
                          tol_mf, tol_smb, tol_hml, use_turnover, turnover_cap):
    tc_str = (f"{float(turnover_cap or 0.30)*100:.0f}%"
              if use_turnover and "on" in use_turnover
              else "None (unconstrained)")
    return dbc.Row([
        dbc.Col(dbc.Alert([
            html.Strong("Backtest Configuration:"),
            html.Ul([
                html.Li(f"Period: {oos_start} for {oos_months} months"),
                html.Li(f"Lookback: {lookback} months"),
                html.Li(f"Rebalance: Every {rebalance_every} months"),
                html.Li(f"Portfolio: {k_max} stocks, max {float(w_max or 0.20)*100:.0f}% per position"),
                html.Li(f"Turnover Cap: {tc_str}"),
            ], className="mb-0"),
        ], color="info"), width=6),
        dbc.Col(dbc.Alert([
            html.Strong("Target Betas:"),
            html.Ul([
                html.Li(f"Market (MF): {target_mf} ± {tol_mf}"),
                html.Li(f"Size (SMB):  {target_smb} ± {tol_smb}"),
                html.Li(f"Value (HML): {target_hml} ± {tol_hml}"),
            ], className="mb-0"),
        ], color="info"), width=6),
    ], className="mb-3")


# ── Run backtest ──────────────────────────────────────────────────────────────
@app.callback(
    Output("backtest-store", "data"),
    Output("backtest-run-output", "children"),
    Input("run-backtest-btn", "n_clicks"),
    State("oos-start", "value"),
    State("oos-months", "value"),
    State("lookback-months", "value"),
    State("rebalance-every", "value"),
    State("k-max", "value"),
    State("w-max", "value"),
    State("risk-aversion", "value"),
    State("target-mf", "value"),
    State("target-smb", "value"),
    State("target-hml", "value"),
    State("tol-mf", "value"),
    State("tol-smb", "value"),
    State("tol-hml", "value"),
    State("use-turnover-cap", "value"),
    State("turnover-cap", "value"),
    State("initial-capital", "value"),
    prevent_initial_call=True,
)
def run_backtest(_, oos_start, oos_months, lookback, rebalance_every,
                 k_max, w_max, risk_aversion,
                 target_mf, target_smb, target_hml,
                 tol_mf, tol_smb, tol_hml,
                 use_turnover, turnover_cap, initial_capital):
    from utils import backtest_fixed_window_quarterly_rebalance_on_breach

    target_betas    = {"MF": float(target_mf or 1.0), "SMB": float(target_smb or 0.0), "HML": float(target_hml or 0.2)}
    beta_tolerances = {"MF": float(tol_mf or 0.3),   "SMB": float(tol_smb or 0.3),   "HML": float(tol_hml or 0.3)}
    tc              = float(turnover_cap or 0.30) if use_turnover and "on" in use_turnover else None

    try:
        bt = backtest_fixed_window_quarterly_rebalance_on_breach(
            stock_returns_data=stock_returns_data,
            fama_french_data=fama_french_data,
            index_returns=index_returns_data,
            universe_by_year=yearly_tickers_data,
            oos_start=oos_start,
            oos_months=int(oos_months or 24),
            lookback_months=int(lookback or 36),
            rebalance_every=int(rebalance_every or 3),
            initial_capital=float(initial_capital or 100000),
            risk_aversion=float(risk_aversion or 1.0),
            K_max=int(k_max or 15),
            w_max=float(w_max or 0.20),
            target_betas=target_betas,
            beta_tolerances=beta_tolerances,
            turnover_cap=tc,
            show_progress=False,
        )

        store = {
            "strategy_value":  bt["strategy_value"].to_json(date_format="iso"),
            "index_value":     bt["index_value"].to_json(date_format="iso"),
            "initial_capital": float(initial_capital or 100000),
            "oos_months":      int(oos_months or 24),
        }
        msg = dbc.Alert("✅ Backtest complete! Switch to the Results tab.", color="success")
        return store, msg

    except Exception as exc:
        return None, dbc.Alert(f"❌ Backtest failed: {exc}", color="danger")


# ── Results display ────────────────────────────────────────────────────────────
@app.callback(
    Output("results-content", "children"),
    Input("backtest-store", "data"),
)
def show_results(store):
    if not store:
        return dbc.Alert("👈 Run a backtest first in the 'Run Backtest' tab.", color="secondary")

    initial_capital = float(store.get("initial_capital", 100000))
    oos_months      = int(store.get("oos_months", 24))

    strategy_value = pd.read_json(store["strategy_value"], typ="series")
    index_value    = pd.read_json(store["index_value"],    typ="series")
    strategy_value = strategy_value.sort_index()
    index_value    = index_value.sort_index()

    final_strat   = strategy_value.iloc[-1]
    final_idx     = index_value.iloc[-1]
    strat_ret_pct = (final_strat - initial_capital) / initial_capital * 100
    idx_ret_pct   = (final_idx   - initial_capital) / initial_capital * 100
    outperf_pct   = (final_strat - final_idx) / final_idx * 100
    ann_vol_pct   = strategy_value.pct_change().std() * np.sqrt(12) * 100

    def _color(v): return "text-success" if v >= 0 else "text-danger"

    metrics = dbc.Row([
        dbc.Col(_kpi_card("Customized Portfolio",   f"${final_strat:,.0f}",  f"{strat_ret_pct:+.2f}%",  _color(strat_ret_pct)), width=3),
        dbc.Col(_kpi_card("Nifty 50",               f"${final_idx:,.0f}",    f"{idx_ret_pct:+.2f}%",    _color(idx_ret_pct)),   width=3),
        dbc.Col(_kpi_card("Outperformance",         f"{outperf_pct:+.2f}%",  f"${final_strat-final_idx:+,.0f}", _color(outperf_pct)), width=3),
        dbc.Col(_kpi_card("Annual Volatility",      f"{ann_vol_pct:.2f}%",   "Customized Portfolio",   "text-info"),            width=3),
    ], className="mb-4 g-2")

    # Portfolio value
    fig_val = go.Figure([
        go.Scatter(x=strategy_value.index, y=strategy_value.values,
                   name="Customized Portfolio", line=dict(color="#1f77b4", width=2)),
        go.Scatter(x=index_value.index, y=index_value.values,
                   name="Nifty 50", line=dict(color="#ff7f0e", width=2)),
    ])
    fig_val.update_layout(template="plotly_dark", title="Portfolio Value Over Time",
                          xaxis_title="Date", yaxis_title="Value ($)", height=420)

    # Monthly returns
    sm = strategy_value.pct_change().dropna()
    im = index_value.pct_change().dropna()
    fig_ret = go.Figure([
        go.Bar(x=sm.index, y=sm.values, name="Customized Portfolio",
               marker_color="#1f77b4", opacity=0.75),
        go.Bar(x=im.index, y=im.values, name="Nifty 50",
               marker_color="#ff7f0e", opacity=0.75),
    ])
    fig_ret.update_layout(template="plotly_dark", title="Monthly Returns",
                          xaxis_title="Month", yaxis_title="Return",
                          barmode="overlay", height=320)

    return html.Div([metrics, dcc.Graph(figure=fig_val), dcc.Graph(figure=fig_ret)])


def _kpi_card(title, main_val, sub_val, sub_class):
    return dbc.Card(dbc.CardBody([
        html.H6(title, className="card-subtitle text-muted mb-1"),
        html.H4(main_val, className="card-title mb-1"),
        html.P(sub_val, className=f"{sub_class} small mb-0"),
    ]), color="dark", outline=True)


# ── Risk sensitivity ───────────────────────────────────────────────────────────
@app.callback(
    Output("sensitivity-output", "children"),
    Input("run-sensitivity-btn", "n_clicks"),
    State("min-ra", "value"),
    State("max-ra", "value"),
    State("num-scenarios", "value"),
    State("oos-start", "value"),
    State("oos-months", "value"),
    State("lookback-months", "value"),
    State("rebalance-every", "value"),
    State("k-max", "value"),
    State("w-max", "value"),
    State("target-mf", "value"),
    State("target-smb", "value"),
    State("target-hml", "value"),
    State("tol-mf", "value"),
    State("tol-smb", "value"),
    State("tol-hml", "value"),
    State("use-turnover-cap", "value"),
    State("turnover-cap", "value"),
    State("initial-capital", "value"),
    prevent_initial_call=True,
)
def run_sensitivity(_, min_ra, max_ra, num_scenarios,
                    oos_start, oos_months, lookback, rebalance_every,
                    k_max, w_max, target_mf, target_smb, target_hml,
                    tol_mf, tol_smb, tol_hml,
                    use_turnover, turnover_cap, initial_capital):
    from utils import backtest_fixed_window_quarterly_rebalance_on_breach

    target_betas    = {"MF": float(target_mf or 1.0), "SMB": float(target_smb or 0.0), "HML": float(target_hml or 0.2)}
    beta_tolerances = {"MF": float(tol_mf or 0.3),   "SMB": float(tol_smb or 0.3),   "HML": float(tol_hml or 0.3)}
    tc              = float(turnover_cap or 0.30) if use_turnover and "on" in use_turnover else None
    initial_cap     = float(initial_capital or 100000)
    oos_m           = int(oos_months or 24)

    ra_values  = np.linspace(float(min_ra or 1.0), float(max_ra or 5.0), int(num_scenarios or 5))
    results    = {}

    for ra in ra_values:
        try:
            bt = backtest_fixed_window_quarterly_rebalance_on_breach(
                stock_returns_data=stock_returns_data,
                fama_french_data=fama_french_data,
                index_returns=index_returns_data,
                universe_by_year=yearly_tickers_data,
                oos_start=oos_start,
                oos_months=oos_m,
                lookback_months=int(lookback or 36),
                rebalance_every=int(rebalance_every or 3),
                initial_capital=initial_cap,
                risk_aversion=float(ra),
                K_max=int(k_max or 15),
                w_max=float(w_max or 0.20),
                target_betas=target_betas,
                beta_tolerances=beta_tolerances,
                turnover_cap=tc,
                show_progress=False,
            )
            results[round(ra, 2)] = bt
        except Exception:
            pass

    if not results:
        return dbc.Alert("All scenarios failed. Check your parameters.", color="danger")

    fig = go.Figure()
    for ra, bt in sorted(results.items()):
        fig.add_trace(go.Scatter(
            x=bt["strategy_value"].index, y=bt["strategy_value"].values,
            name=f"RA={ra:.2f}", line=dict(width=2),
        ))
    fig.update_layout(
        template="plotly_dark",
        title="Portfolio Performance Across Risk Aversion Levels",
        xaxis_title="Date", yaxis_title="Portfolio Value ($)", height=450,
    )

    summary_rows = []
    for ra, bt in sorted(results.items()):
        fv = bt["strategy_value"].iloc[-1]
        tr = (fv - initial_cap) / initial_cap * 100
        ar = tr / (oos_m / 12)
        av = bt["strategy_value"].pct_change().std() * np.sqrt(12) * 100
        summary_rows.append({
            "Risk Aversion": f"{ra:.2f}",
            "Final Value":   f"${fv:,.0f}",
            "Total Return":  f"{tr:.2f}%",
            "Ann. Return":   f"{ar:.2f}%",
            "Ann. Vol":      f"{av:.2f}%",
        })

    table = dbc.Table.from_dataframe(
        pd.DataFrame(summary_rows),
        striped=True, bordered=True, hover=True, color="dark",
    )

    return html.Div([
        dbc.Alert(f"✅ Completed {len(results)} scenarios", color="success"),
        dcc.Graph(figure=fig),
        html.H5("Summary Statistics", className="mt-3"),
        table,
    ])


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)
