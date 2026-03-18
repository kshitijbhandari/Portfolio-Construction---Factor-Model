# Factor Model Portfolio Optimizer

A Streamlit web app for Fama-French 3-factor portfolio optimization and backtesting on the Nifty 50 universe.

---

## Features

- **Beta Explorer** — Visualize MF/SMB/HML factor returns over the lookback window before committing to target betas; compute achievable beta ranges via MILP
- **Sector Dynamics** — Equal-weighted sector performance dashboard: cumulative return trend charts + metrics table (12M/3M return, momentum, volatility, positive months, MF beta)
- **Portfolio Optimization** — MILP (PuLP/CBC) with target factor exposures, cardinality constraint, position size limits, optional turnover cap, and sector-level stock count constraints
- **Backtest** — Rolling-window rebalance with breach-triggered rebalancing; portfolio vs Nifty 50 comparison
- **Results** — Portfolio value chart, monthly returns, rebalancing log, composition table
- **Risk Analysis** — Sensitivity sweep across risk aversion values

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Local app opens at **http://localhost:8501**

**Live app**: https://huggingface.co/spaces/kshitijbhandari/Factor-based-Portfolio-Constructor

---

## Stock Universe Construction

The investment universe is the **historical NIFTY 50** — the exact set of stocks that were constituents of the index at any given point in time, going back to June 2006.

### Why Historical Composition?
Using only the *current* NIFTY 50 for backtesting would introduce **survivorship bias** — stocks that were dropped (due to bankruptcy, delisting, poor performance) would be excluded, making past performance look artificially better. Instead, the model uses point-in-time composition snapshots.

### How It Was Built

1. **Scraped Wikipedia** for the current NIFTY 50 constituents and a full log of every index addition/removal since June 2006 (64 change events across 44 unique dates)

2. **Built a ticker mapping** — Wikipedia uses company names; a lookup table (`nifty_50_historical_companies.csv`) maps all 88 current and historical constituents to their NSE ticker symbols

3. **Backward reconstruction** — Starting from today's known 50 stocks, the algorithm walks backwards through the change log:
   - At each change date: remove newly added stocks, add back removed ones
   - This gives the exact index composition just before that change
   - Repeated across all 44 change events back to June 2006

4. **Output**: `Nifty_50.csv` — a 44 × 50 matrix where each row is a change date and each column is a ticker in the NIFTY 50 at that time

The backtest engine uses this file to select the eligible stock universe at each rebalance date, ensuring only stocks that were actually in the index at that time are considered.

---

## File Structure

```
Personal_Factor_model/
├── app.py                    # Streamlit app
├── utils.py                  # Core optimization & backtest functions
├── setup_streamlit.py        # Auto-setup helper
├── requirements.txt
├── model.ipynb               # Original research notebook (gitignored)
└── data/
    ├── nifty_stocks_data (1).csv   # Monthly stock returns (Date, Ticker, RET)
    ├── nifty50_index_data.csv      # Nifty 50 index returns
    ├── FF_Nifty50.csv              # Fama-French factors (MF, SMB, HML, RF)
    ├── Nifty_50.csv                # Universe tickers by year
    └── sector_classification.csv   # Ticker → sector mapping
```

---

## Tab Guide

### 📐 Beta Explorer
- **Factor trend charts** — 3 stacked panels (MF, SMB, HML) showing monthly factor returns as colour-coded bars + 3-month rolling average over the selected lookback window
- **Factor statistics table** — annualised mean, vol, Sharpe, min/max
- **Achievable beta ranges** — click "Compute Beta Ranges" to run a MILP that finds the min/max portfolio beta reachable with your K_max and w_max constraints; checks whether your target betas are feasible before running the backtest

### 🏭 Sector Dynamics
Click **▶️ Compute Sector Dynamics** to generate:

**Metrics table** (sectors as columns):

| Metric | Description |
|--------|-------------|
| 12M Return | Cumulative equal-weighted sector return over last 12 months |
| 3M Return | Cumulative return over last 3 months |
| Momentum | Accelerating (green) if 3M > 12M, else Decelerating (red) |
| Ann. Volatility | Monthly std × √12 over full lookback |
| Positive Months | Count of months with positive sector return |
| MF Beta | OLS regression of sector returns vs MF factor |

**Trend charts** — small cumulative return panels (3 per row), one per sector. Green fill = positive cumulative return, red fill = negative.

Results are cached; use **🔄 Recompute** after changing the as-of date or lookback period.

### 📊 Run Backtest
Configure parameters in the sidebar and click **▶️ Run Backtest**.

### 📈 Results
- Final value, total return, outperformance vs Nifty 50, annual volatility
- Portfolio value chart and monthly returns bar chart
- Rebalancing log with factor exposures at each rebalance date
- Portfolio composition table (weights % by stock and period)

### 🔍 Risk Analysis
Run N backtests across a range of risk aversion values; compare portfolio value curves and summary statistics.

### ℹ️ Info
Model documentation, optimization formulation, data requirements.

---

## Sidebar Configuration

### Backtest Parameters

| Parameter | Range | Default |
|-----------|-------|---------|
| Out-of-Sample Start | any month in data | ~5 years before latest |
| OOS Duration | 6–48 months | 24 |
| Lookback Period | 12–120 months | 36 |
| Rebalance Frequency | 1–12 months | 3 |

### Optimization Constraints

| Parameter | Range | Default |
|-----------|-------|---------|
| Max Positions (K_max) | 5–50 | 15 |
| Max Position Size (w_max) | 5%–50% | 20% |
| Risk Aversion (λ) | 0.1–10 | 1.0 |

### Target Betas & Tolerances

| Factor | Default Target | Default Tolerance |
|--------|---------------|-------------------|
| MF (Market) | 1.0 | ±0.3 |
| SMB (Size) | 0.0 | ±0.3 |
| HML (Value) | 0.2 | ±0.3 |

### Turnover Cap
Optional constraint limiting total portfolio turnover per rebalance (`|w_new - w_old|` summed across all positions).

### Sector Constraints
Enable per-sector stock count limits:
- Tick a sector checkbox to activate its constraint
- Set **min** (≥ N stocks from this sector) and/or **max** (≤ N stocks)
- **Set max = 0 to fully exclude a sector** from the portfolio
- Click **✅ Confirm Sector Constraints** to apply all inputs at once — no per-keystroke rerenders

---

## Model Summary

**Regression (per stock, rolling window):**
```
RET - RF = α + β_MF·MF + β_SMB·SMB + β_HML·HML + ε
```

**Optimization (MILP via PuLP/CBC):**
```
maximize  μᵀw - λ·MAD(Rₚ)
subject to
  Σwᵢ = 1                          fully invested
  wᵢ ≤ w_max · zᵢ                  position size limit
  Σzᵢ ≤ K_max                      cardinality
  β_target - tol ≤ Bw ≤ β_target + tol   factor exposure bands
  Σ_{i∈sector} zᵢ ∈ [min_k, max_k]       sector constraints (optional)
  Σ|wᵢ - wᵢ_prev| ≤ turnover_cap         turnover cap (optional)
```

---

## Troubleshooting

**Optimization infeasible** — Use Beta Explorer → Compute Beta Ranges to see what betas are actually achievable. Widen tolerances or adjust targets accordingly.

**Data not found** — Ensure all CSV files are present in the `data/` directory including `sector_classification.csv` (columns: `company`, `sector`).

**App slow** — Reduce OOS duration, increase rebalance frequency, or reduce K_max.

---

## Dependencies

Streamlit · PuLP · Pandas · NumPy · Matplotlib · SciPy · tqdm

---

**Version**: 2.0 | **Last Updated**: March 2026
