import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from datetime import datetime
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

# ============================================================================
# MODULE-LEVEL CACHED FUNCTIONS
# ============================================================================
@st.cache_data(max_entries=10)
def compute_sector_dynamics(oos_start, lookback_months, _stock_returns, _fama_french, _ticker_to_sector):
    asof_dt = pd.Period(oos_start, "M").to_timestamp(how="end")
    lb_dt   = asof_dt - pd.DateOffset(months=lookback_months)

    sr = _stock_returns
    sr_win = sr[(sr["Date"] > lb_dt) & (sr["Date"] <= asof_dt)]
    sr_win = sr_win.assign(sector=sr_win["Ticker"].map(_ticker_to_sector))
    sr_win = sr_win.dropna(subset=["sector"])

    sector_monthly = (
        sr_win.groupby(["Date", "sector"])["RET"]
        .mean()
        .unstack("sector")
        .sort_index()
    )
    sector_monthly.index = pd.to_datetime(sector_monthly.index)

    mf_series = None
    if "MF" in _fama_french.columns:
        ff_tmp = _fama_french.copy()
        ff_tmp["Date"] = pd.to_datetime(ff_tmp["Date"])
        mf_series = ff_tmp.set_index("Date")["MF"]

    metrics = {}
    for sec in sorted(sector_monthly.columns):
        s = sector_monthly[sec].dropna().sort_index()
        if len(s) < 2:
            continue

        s_12 = s.iloc[-12:] if len(s) >= 12 else s
        s_3  = s.iloc[-3:]  if len(s) >= 3  else s

        ret_12 = float((1 + s_12).prod() - 1)
        ret_3  = float((1 + s_3).prod() - 1)
        momentum  = "Accelerating" if ret_3 > ret_12 else "Decelerating"
        ann_vol   = float(s.std() * np.sqrt(12))
        pos_months = int((s > 0).sum())

        mf_beta = np.nan
        if mf_series is not None:
            common = s.index.intersection(mf_series.index)
            if len(common) >= 6:
                y = s.loc[common].values
                x = mf_series.loc[common].values
                X = np.column_stack([np.ones(len(x)), x])
                try:
                    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                    mf_beta = float(coef[1])
                except Exception:
                    pass

        metrics[sec] = {
            "12M Return":      f"{ret_12:+.1%}",
            "3M Return":       f"{ret_3:+.1%}",
            "Momentum":        momentum,
            "Ann. Volatility": f"{ann_vol:.1%}",
            "Positive Months": pos_months,
            "MF Beta":         f"{mf_beta:.2f}" if not np.isnan(mf_beta) else "N/A",
        }

    sectors_sorted = sorted(sector_monthly.columns.tolist())
    n_sectors = len(sectors_sorted)
    n_cols    = 3
    n_rows    = (n_sectors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 2.8))
    axes_flat = np.array(axes).flatten()

    for i, sec in enumerate(sectors_sorted):
        ax  = axes_flat[i]
        s   = sector_monthly[sec].dropna().sort_index()
        cum = (1 + s).cumprod() - 1

        final_val  = float(cum.iloc[-1]) if len(cum) else 0
        line_color = "#2ca02c" if final_val >= 0 else "#d62728"
        fill_color = "#c8e6c9" if final_val >= 0 else "#ffcdd2"

        x = list(range(len(cum)))
        ax.plot(x, cum.values * 100, color=line_color, linewidth=1.3)
        ax.fill_between(x, 0, cum.values * 100, alpha=0.25, color=fill_color)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(sec, fontsize=8, fontweight="bold", pad=2)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(True, alpha=0.2, axis="y")
        ax.set_xticks([])

    for j in range(n_sectors, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout(pad=1.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)

    return sector_monthly, metrics, buf.getvalue()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Factor Model Portfolio Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Fama-French Factor Model Portfolio Optimizer")
st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    # ── Strategy mode ─────────────────────────────────────────────────────────
    strategy_mode = st.radio(
        "Strategy Mode",
        ["Customized Strategy", "Recommended Strategy"],
        horizontal=True,
    )
    is_recommended = strategy_mode == "Recommended Strategy"

    st.divider()

    # ── Data loading ──────────────────────────────────────────────────────────
    st.subheader("📁 Data Files")

    if os.path.exists("data/nifty_stocks_data_clean.csv"):
        data_dir = "data"
        st.info("✅ Using local data files")
    elif os.path.exists("c:\\Users\\kshit\\Personal_Factor_model\\data\\nifty_stocks_data_clean.csv"):
        data_dir = "c:\\Users\\kshit\\Personal_Factor_model\\data"
        st.info("✅ Using local absolute path")
    else:
        data_dir = st.text_input("Data Directory", value="data")

    try:
        @st.cache_data
        def load_data(data_dir):
            data_dir = os.path.normpath(data_dir)
            stock_returns = pd.read_csv(os.path.join(data_dir, 'nifty_stocks_data_clean.csv'))
            index_returns = pd.read_csv(os.path.join(data_dir, 'nifty50_index_data.csv'))
            fama_french   = pd.read_csv(os.path.join(data_dir, 'FF_Nifty50.csv'))
            sector_df     = pd.read_csv(os.path.join(data_dir, 'sector_classification.csv'))
            return stock_returns, index_returns, fama_french, sector_df

        stock_returns_data, index_returns_data, fama_french_data, sector_data = load_data(data_dir)
        ticker_to_sector = dict(zip(sector_data['company'], sector_data['sector']))
        all_sectors = sorted(sector_data['sector'].unique().tolist())
        st.success("✅ Data loaded successfully")

    except FileNotFoundError as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

    @st.cache_data
    def _build_universe(stock_returns, lookback, min_obs_val):
        from utils import build_R_full, build_ticker_universe
        r_full = build_R_full(stock_returns)
        start = r_full.index.min().strftime('%Y-%m-%d')
        end   = r_full.index.max().strftime('%Y-%m-%d')
        t_uni = build_ticker_universe(r_full, start, end, lookback_months=lookback, min_obs=min_obs_val)
        return r_full, t_uni

    st.divider()

    # ── Backtest parameters ───────────────────────────────────────────────────
    st.subheader("📈 Backtest Parameters")

    stock_returns_data["Date"] = pd.to_datetime(stock_returns_data["Date"])
    min_date = stock_returns_data["Date"].min()
    max_date = stock_returns_data["Date"].max()
    all_months = pd.period_range(start=min_date, end=max_date, freq='M')
    oos_options = [str(m) for m in all_months]

    oos_start = st.selectbox(
        "Out-of-Sample Start",
        options=oos_options,
        index=max(0, len(oos_options) - 60),
        help="Start month for backtest",
    )

    oos_months = st.slider(
        "OOS Duration (months)",
        min_value=3, max_value=48, value=24, step=1,
    )

    if is_recommended:
        # Strategy dropdown
        strategy_labels = [
            "1. Best betas strategy",
            "2. Mean Lookbck period strategy",
            "3. Median lookback period strategy",
            "4. Monthly mean best betas strategy",
            "5. Mean past best betas (6,6,1) strategy",
            "Strategy 6 mean of regime best betas",
        ]
        strategy_choice = st.selectbox(
            "Strategy",
            strategy_labels,
            help=(
                "1 = best betas, 2 = mean lookback period, 3 = median lookback period, "
                "4 = monthly mean best betas, 5 = mean past best betas (6,6,1), "
                "6 = mean of regime best betas"
            ),
        )
        beta_source_map = {
            "1. Best betas strategy": "best",
            "2. Mean Lookbck period strategy": "mean",
            "3. Median lookback period strategy": "median",
            "4. Monthly mean best betas strategy": "monthly_mean",
            "5. Mean past best betas (6,6,1) strategy": "mean_past_best_661",
            "Strategy 6 mean of regime best betas": "regime_mean",
        }
        beta_source = beta_source_map[strategy_choice]

        # Fixed defaults for recommended mode
        lookback_months  = 36
        rebalance_every  = 3
        K_max            = 15
        w_max            = 0.20
        risk_aversion    = 1.0
        target_betas     = None
        beta_tolerances  = {"MF": 0.30, "SMB": 0.30, "HML": 0.30}
        turnover_cap     = 0.20
        sector_constraints = None
        initial_capital  = 100_000

    else:
        # Customized: full controls
        lookback_months = st.slider(
            "Lookback Period (months)",
            min_value=12, max_value=120, value=36, step=12,
        )

        rebalance_every = st.slider(
            "Rebalance Frequency (months)",
            min_value=1, max_value=12, value=3, step=1,
        )

        st.divider()
        st.subheader("🎯 Optimization Constraints")

        col1, col2 = st.columns(2)
        with col1:
            K_max = st.slider("Max Positions", min_value=5, max_value=50, value=15, step=1)
        with col2:
            w_max = st.slider("Max Position Size", min_value=0.05, max_value=0.5, value=0.20, step=0.05)

        risk_aversion = st.slider("Risk Aversion", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

        st.divider()
        st.subheader("📊 Target Betas & Tolerances")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Target Betas**")
            target_mf  = st.number_input("Market Beta (MF)",  value=1.0, step=0.1)
            target_smb = st.number_input("Size Beta (SMB)",   value=0.0, step=0.1)
            target_hml = st.number_input("Value Beta (HML)",  value=0.2, step=0.1)
        with col2:
            st.write("**Tolerances (±)**")
            tol_mf  = st.number_input("MF Tolerance",  value=0.3, step=0.05)
            tol_smb = st.number_input("SMB Tolerance", value=0.3, step=0.05)
            tol_hml = st.number_input("HML Tolerance", value=0.3, step=0.05)

        target_betas     = {"MF": target_mf, "SMB": target_smb, "HML": target_hml}
        beta_tolerances  = {"MF": tol_mf,    "SMB": tol_smb,    "HML": tol_hml}

        st.divider()
        st.subheader("🔁 Turnover Constraint")
        use_turnover_cap = st.checkbox("Enable Turnover Cap", value=False)
        if use_turnover_cap:
            turnover_cap = st.slider("Turnover Cap", min_value=0.05, max_value=1.0, value=0.30, step=0.05)
        else:
            turnover_cap = None

        st.divider()
        st.subheader("🏭 Sector Constraints")
        use_sector_constraints = st.checkbox("Enable Sector Constraints", value=False)

        sector_constraints = st.session_state.get("confirmed_sector_constraints", None)
        if use_sector_constraints:
            st.caption("Set limits, then click **Confirm** to apply.")
            with st.form("sector_constraints_form"):
                raw = {}
                for sec in all_sectors:
                    col_a, col_b, col_c = st.columns([2, 1, 1])
                    with col_a:
                        enabled = st.checkbox(sec, value=False, key=f"sec_enabled_{sec}")
                    with col_b:
                        mn = st.number_input("min", min_value=0, max_value=20, value=0, step=1, key=f"sec_min_{sec}")
                    with col_c:
                        mx = st.number_input("max", min_value=0, max_value=20, value=5, step=1, key=f"sec_max_{sec}")
                    raw[sec] = (enabled, mn, mx)

                submitted = st.form_submit_button("✅ Confirm Sector Constraints", use_container_width=True, type="primary")
                if submitted:
                    confirmed = {sec: (mn if mn > 0 else None, mx) for sec, (enabled, mn, mx) in raw.items() if enabled}
                    st.session_state["confirmed_sector_constraints"] = confirmed if confirmed else None
                    sector_constraints = st.session_state["confirmed_sector_constraints"]
        else:
            st.session_state.pop("confirmed_sector_constraints", None)
            sector_constraints = None

        st.divider()
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=10000, max_value=10000000, value=100000, step=10000,
        )
        beta_source = None  # not used in customized mode

    # Build universe
    _lb = lookback_months
    with st.spinner("Building ticker universe..."):
        R_full, ticker_universe = _build_universe(stock_returns_data, _lb, _lb)

# ============================================================================
# TABS
# ============================================================================
tab4, tab6, tab1, tab2, tab3, tab5 = st.tabs([
    "📐 Beta Explorer", "🏭 Sector Dynamics",
    "📊 Run Backtest", "📈 Results",
    "🔍 Risk Analysis", "ℹ️ Info",
])

# ============================================================================
# TAB: SECTOR DYNAMICS
# ============================================================================
with tab6:
    st.header("Sector Dynamics")

    _sd_key = f"{oos_start}|{lookback_months}"
    if st.session_state.get("sd_key") != _sd_key:
        st.session_state.pop("sd_results", None)

    if "sd_results" not in st.session_state:
        st.info(f"As-of: **{oos_start}**, lookback: **{lookback_months}m**")
        if st.button("▶️ Compute Sector Dynamics", type="primary", use_container_width=True):
            with st.spinner("Computing sector metrics..."):
                try:
                    sector_monthly, metrics, fig_bytes = compute_sector_dynamics(
                        oos_start, lookback_months,
                        stock_returns_data, fama_french_data, ticker_to_sector,
                    )
                    st.session_state["sd_results"] = (sector_monthly, metrics, fig_bytes)
                    st.session_state["sd_key"] = _sd_key
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.exception(e)
    else:
        sector_monthly, metrics, fig_bytes = st.session_state["sd_results"]
        st.caption(f"Equal-weighted sector performance — **{lookback_months}-month** lookback ending **{oos_start}**.")
        if st.button("🔄 Recompute"):
            st.session_state.pop("sd_results", None)
            st.rerun()

        st.subheader("📊 Sector Metrics Table")
        metrics_df = pd.DataFrame(metrics)

        def _style_cell(val):
            if val == "Accelerating":
                return "background-color: #d4edda; color: #155724; font-weight: bold"
            if val == "Decelerating":
                return "background-color: #f8d7da; color: #721c24; font-weight: bold"
            return ""

        try:
            styled = metrics_df.style.map(_style_cell)
        except AttributeError:
            styled = metrics_df.style.applymap(_style_cell)

        st.dataframe(styled, use_container_width=True)
        st.divider()
        st.subheader("📈 Cumulative Return Trends")
        st.image(fig_bytes, use_container_width=True)

# ============================================================================
# TAB: RUN BACKTEST
# ============================================================================
with tab1:
    st.header("Run Portfolio Backtest")

    if is_recommended:
        # ── Recommended mode info ─────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""
**Recommended Strategy — {strategy_choice}**
- Period: {oos_start} for {oos_months} months
- Rebalance every {rebalance_every} months
- Lookback: {lookback_months} months
- Portfolio: {K_max} stocks, max {w_max*100:.0f}% per position
            """)
        with col2:
            st.info(f"""
**Beta source:** {beta_source} betas from Excel log

Betas are pulled from `beta_search_log.xlsx` at each rebalance date.
The optimizer targets those betas automatically.
            """)

        # Resolve excel source: local file or uploaded
        _local_candidates = [
            os.path.join(data_dir, "beta_search_log.xlsx"),
            "beta_search_log.xlsx",
            os.path.join(os.path.dirname(__file__), "beta_search_log.xlsx"),
            os.path.join(os.path.dirname(__file__), "data", "beta_search_log.xlsx"),
        ]
        _local_path = next((p for p in _local_candidates if os.path.exists(p)), None)

        if _local_path:
            _excel_source = _local_path
            st.success("✅ `beta_search_log.xlsx` found locally.")
        else:
            st.warning("⚠️ `beta_search_log.xlsx` not found. Upload it below.")
            _uploaded = st.file_uploader(
                "Upload beta_search_log.xlsx",
                type=["xlsx"],
                help="Generated by running the Optuna beta search in model.ipynb",
            )
            _excel_source = _uploaded  # None if not uploaded yet

        if st.button("▶️ Run Recommended Backtest", use_container_width=True, type="primary"):
            if _excel_source is None:
                st.error("Upload `beta_search_log.xlsx` before running.")
            else:
                with st.spinner(f"⏳ Running {strategy_choice} backtest from {oos_start}..."):
                    try:
                        from utils import run_recommended_backtest
                        backtest_result = run_recommended_backtest(
                            stock_returns_data=stock_returns_data,
                            fama_french_data=fama_french_data,
                            index_returns=index_returns_data,
                            oos_start=oos_start,
                            oos_months=oos_months,
                            beta_source=beta_source,
                            rebalance_every=rebalance_every,
                            risk_aversion=risk_aversion,
                            lookback_months=lookback_months,
                            K_max=K_max,
                            w_max=w_max,
                            turnover_cap=turnover_cap,
                            initial_capital=initial_capital,
                            excel_path=_excel_source,
                            R_full_prebuilt=R_full,
                            ticker_universe_prebuilt=ticker_universe,
                        )
                        st.session_state.backtest_result = backtest_result
                        st.session_state.backtest_mode   = "recommended"
                        st.session_state.strategy_label  = strategy_choice
                        st.success(f"✅ {strategy_choice} backtest completed!")
                    except Exception as e:
                        st.error(f"❌ Backtest failed: {str(e)}")
                        st.exception(e)

    else:
        # ── Customized mode info ──────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""
**Backtest Configuration:**
- Period: {oos_start} for {oos_months} months
- Lookback: {lookback_months} months
- Rebalance: Every {rebalance_every} months
- Portfolio: {K_max} stocks, max {w_max*100:.0f}% per position
- Turnover Cap: {f"{turnover_cap*100:.0f}%" if turnover_cap is not None else "None"}
            """)
        with col2:
            st.info(f"""
**Target Betas:**
- Market (MF): {target_betas['MF']} ± {beta_tolerances['MF']}
- Size (SMB): {target_betas['SMB']} ± {beta_tolerances['SMB']}
- Value (HML): {target_betas['HML']} ± {beta_tolerances['HML']}
            """)

        if st.button("▶️ Run Backtest", use_container_width=True, type="primary"):
            backtest_func = None
            try:
                from utils import backtest_fixed_window_quarterly_rebalance_on_breach
                backtest_func = backtest_fixed_window_quarterly_rebalance_on_breach
            except Exception as e:
                st.error(f"❌ Cannot import backtest function: {e}")
                st.stop()

            with st.spinner(f"⏳ Running backtest from {oos_start}..."):
                try:
                    backtest_result = backtest_func(
                        stock_returns_data=stock_returns_data,
                        fama_french_data=fama_french_data,
                        index_returns=index_returns_data,
                        ticker_universe=ticker_universe,
                        oos_start=oos_start,
                        oos_months=oos_months,
                        lookback_months=lookback_months,
                        rebalance_every=rebalance_every,
                        initial_capital=initial_capital,
                        risk_aversion=risk_aversion,
                        K_max=K_max,
                        w_max=w_max,
                        target_betas=target_betas,
                        beta_tolerances=beta_tolerances,
                        turnover_cap=turnover_cap,
                        sector_constraints=sector_constraints,
                        ticker_to_sector=ticker_to_sector if sector_constraints else None,
                        show_progress=True,
                    )
                    st.session_state.backtest_result = backtest_result
                    st.session_state.backtest_mode   = "customized"
                    st.session_state.strategy_label  = "Customized Strategy"
                    st.success("✅ Backtest completed successfully!")
                except Exception as e:
                    st.error(f"❌ Backtest failed: {str(e)}")
                    st.exception(e)

# ============================================================================
# TAB: RESULTS
# ============================================================================
with tab2:
    st.header("Backtest Results")

    if "backtest_result" not in st.session_state:
        st.info("👈 Run a backtest first in the 'Run Backtest' tab")
    else:
        bt = st.session_state.backtest_result
        _mode  = st.session_state.get("backtest_mode", "customized")
        _label = st.session_state.get("strategy_label", "Portfolio")

        strategy_value = bt["strategy_value"]
        index_value    = bt["index_value"]
        _init_cap      = float(strategy_value.iloc[0] / (1 + bt["strategy_returns"].iloc[0]))

        # ── Key metrics ───────────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)

        final_strategy  = strategy_value.iloc[-1]
        final_index     = index_value.iloc[-1]
        strategy_return = (final_strategy - _init_cap) / _init_cap * 100
        index_return    = (final_index    - _init_cap) / _init_cap * 100

        with col1:
            st.metric(f"{_label} Final Value", f"${final_strategy:,.0f}", f"{strategy_return:+.2f}%")
        with col2:
            st.metric("Nifty 50 Final Value",  f"${final_index:,.0f}",   f"{index_return:+.2f}%")
        with col3:
            outperformance = (final_strategy - final_index) / final_index * 100
            st.metric("Outperformance vs Nifty 50", f"{outperformance:+.2f}%", f"${final_strategy - final_index:+,.0f}")
        with col4:
            ann_vol = bt["strategy_returns"].std() * np.sqrt(12)
            st.metric("Annual Volatility", f"{ann_vol:.2%}")

        st.divider()

        # ── Portfolio value chart ─────────────────────────────────────────────
        st.subheader("📈 Portfolio Value Over Time")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(strategy_value.index, strategy_value.values, label=_label, linewidth=2, color="#1f77b4")
        ax.plot(index_value.index,    index_value.values,    label="Nifty 50",  linewidth=2, color="#ff7f0e")

        if "rebalance_log" in bt and bt["rebalance_log"]:
            for date_str in bt["rebalance_log"]:
                rb_dt = pd.to_datetime(date_str)
                ax.axvline(rb_dt, color="grey", linestyle=":", linewidth=0.7, alpha=0.6)

        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_title(f"{_label} vs Nifty 50")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)

        # ── Monthly returns ────────────────────────────────────────────────────
        st.subheader("📊 Monthly Returns")
        strategy_monthly = bt["strategy_returns"]
        index_monthly    = bt["index_returns"]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(len(strategy_monthly)), strategy_monthly.values, label=_label, alpha=0.7, color="#1f77b4")
        ax.bar(range(len(index_monthly)),    index_monthly.values,    label="Nifty 50", alpha=0.7, color="#ff7f0e")
        ax.set_xlabel("Month")
        ax.set_ylabel("Return")
        ax.set_title("Monthly Returns Comparison")
        ax.legend()
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)

        # ── Turnover & rebalance log ───────────────────────────────────────────
        if "rebalance_log" in bt and bt["rebalance_log"]:
            rebalance_log = bt["rebalance_log"]

            st.divider()
            st.subheader("🔄 Rebalance & Turnover Log")

            if _mode == "recommended":
                # Recommended mode: log has target_betas, achieved_betas, portfolio_value, turnover_pct, dollar_turnover
                rows = []
                total_dollar = 0.0
                for date_str, info in rebalance_log.items():
                    tb = info.get("target_betas", {})
                    ab = info.get("achieved_betas", {})
                    dt = info.get("dollar_turnover", "initial")
                    tv = info.get("turnover_pct", "initial")
                    pv = info.get("portfolio_value", 0)
                    rows.append({
                        "Date": date_str,
                        "Regime": info.get("regime", "—"),
                        "Portfolio Value": f"${pv:,.0f}",
                        "Turnover": tv,
                        "Dollar Turnover": f"${dt:,.0f}" if isinstance(dt, (int, float)) else str(dt),
                        "Target MF": round(tb.get("MF", 0), 3),
                        "Target SMB": round(tb.get("SMB", 0), 3),
                        "Target HML": round(tb.get("HML", 0), 3),
                        "Achieved MF": ab.get("MF", "—"),
                        "Achieved SMB": ab.get("SMB", "—"),
                        "Achieved HML": ab.get("HML", "—"),
                    })
                    if isinstance(dt, (int, float)):
                        total_dollar += dt

                st.dataframe(pd.DataFrame(rows).set_index("Date"), use_container_width=True)
                st.metric("Total Dollar Trading Volume", f"${total_dollar:,.0f}")

            else:
                # Customized mode: log has within_tolerance, rebalanced, exposure_before/after, weights, portfolio_value, turnover_pct, dollar_turnover
                rows = []
                total_dollar = 0.0
                for date_str, info in rebalance_log.items():
                    did_reb = info.get("rebalanced", False)
                    pv  = info.get("portfolio_value", 0)
                    tv  = info.get("turnover_pct", "—")
                    dt  = info.get("dollar_turnover", 0)
                    exp_b = info.get("exposure_before_or_current", {})
                    exp_a = info.get("exposure_after", {})
                    rows.append({
                        "Date": date_str,
                        "Rebalanced": "✅" if did_reb else "⏭️ Skip",
                        "Portfolio Value": f"${pv:,.0f}",
                        "Turnover": tv if did_reb else "—",
                        "Dollar Turnover": f"${dt:,.0f}" if did_reb and isinstance(dt, (int, float)) else "—",
                        "MF before": round(exp_b.get("MF", 0), 3),
                        "MF after": round(exp_a.get("MF", 0), 3),
                        "SMB before": round(exp_b.get("SMB", 0), 3),
                        "SMB after": round(exp_a.get("SMB", 0), 3),
                        "HML before": round(exp_b.get("HML", 0), 3),
                        "HML after": round(exp_a.get("HML", 0), 3),
                    })
                    if did_reb and isinstance(dt, (int, float)):
                        total_dollar += dt

                st.dataframe(pd.DataFrame(rows).set_index("Date"), use_container_width=True)
                st.metric("Total Dollar Trading Volume", f"${total_dollar:,.0f}")

            st.divider()

            # ── Portfolio composition ──────────────────────────────────────────
            st.subheader("📊 Portfolio Weights at Each Rebalance")
            compositions = {}
            for date_str, info in rebalance_log.items():
                weights = info.get("weights", {})
                if isinstance(weights, pd.Series):
                    weights = weights.to_dict()
                if weights:
                    compositions[date_str] = weights

            if compositions:
                comp_df = pd.DataFrame.from_dict(compositions, orient='index').fillna(0.0)
                comp_pct = (comp_df * 100).round(2)
                st.write("**Weights (%) at each rebalancing date:**")
                st.dataframe(
                    comp_pct.style.format("{:.2f}")
                        .highlight_max(axis=1, color='#FFE5B4')
                        .highlight_min(axis=1, color='#E8F5E9'),
                    use_container_width=True,
                    height=400,
                )

                st.write("**Average position size across all periods:**")
                avg_weights = comp_df.mean().sort_values(ascending=False) * 100
                avg_weights = avg_weights[avg_weights > 0.1]

                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(avg_weights.to_frame("Avg Weight (%)").style.format("{:.2f}"), use_container_width=True)
                with col2:
                    if len(avg_weights) > 0:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.pie(avg_weights, labels=avg_weights.index, autopct='%1.1f%%', startangle=90)
                        ax.set_title("Average Portfolio Allocation")
                        st.pyplot(fig, use_container_width=True)

# ============================================================================
# TAB: RISK ANALYSIS
# ============================================================================
with tab3:
    st.header("Risk Sensitivity Analysis")
    st.info("Run multiple backtests with varying risk aversion parameters")

    if is_recommended:
        st.warning("Switch to **Customized Strategy** mode to use Risk Sensitivity Analysis.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            min_ra = st.slider("Min Risk Aversion", 0.1, 5.0, 1.0, step=0.1)
        with col2:
            max_ra = st.slider("Max Risk Aversion", 1.0, 10.0, 5.0, step=0.1)
        num_scenarios = st.slider("Number of Scenarios", 2, 20, 5, step=1)

        if st.button("🔄 Run Risk Sensitivity Analysis", use_container_width=True, type="primary"):
            try:
                from utils import backtest_fixed_window_quarterly_rebalance_on_breach
                backtest_func = backtest_fixed_window_quarterly_rebalance_on_breach
            except Exception as e:
                st.error(f"Cannot import backtest functions. Error: {e}")
                st.stop()

            risk_aversion_values = np.linspace(min_ra, max_ra, num_scenarios)
            sensitivity_results = {}
            progress_bar = st.progress(0)
            status_text  = st.empty()

            for i, ra in enumerate(risk_aversion_values):
                status_text.text(f"Running scenario {i+1}/{num_scenarios} (RA={ra:.2f})...")
                try:
                    bt_temp = backtest_func(
                        stock_returns_data=stock_returns_data,
                        fama_french_data=fama_french_data,
                        index_returns=index_returns_data,
                        ticker_universe=ticker_universe,
                        oos_start=oos_start,
                        oos_months=oos_months,
                        lookback_months=lookback_months,
                        rebalance_every=rebalance_every,
                        initial_capital=initial_capital,
                        risk_aversion=ra,
                        K_max=K_max,
                        w_max=w_max,
                        target_betas=target_betas,
                        beta_tolerances=beta_tolerances,
                        turnover_cap=turnover_cap,
                        sector_constraints=sector_constraints,
                        ticker_to_sector=ticker_to_sector if sector_constraints else None,
                        show_progress=False,
                    )
                    sensitivity_results[ra] = bt_temp
                except Exception as e:
                    st.warning(f"Scenario RA={ra:.2f} failed: {e}")
                progress_bar.progress((i + 1) / num_scenarios)

            status_text.empty()

            if sensitivity_results:
                st.session_state.sensitivity_results = sensitivity_results
                st.success(f"✅ Completed {len(sensitivity_results)} scenarios")

                fig, ax = plt.subplots(figsize=(12, 6))
                for ra, bt_data in sorted(sensitivity_results.items()):
                    ax.plot(bt_data["strategy_value"].index, bt_data["strategy_value"].values,
                            label=f"RA={ra:.2f}", linewidth=2, alpha=0.8)
                ax.set_xlabel("Date")
                ax.set_ylabel("Portfolio Value ($)")
                ax.set_title("Portfolio Performance Across Risk Aversion Levels")
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig, use_container_width=True)

                summary_data = []
                for ra, bt_data in sorted(sensitivity_results.items()):
                    final_val    = bt_data["strategy_value"].iloc[-1]
                    total_return = (final_val - initial_capital) / initial_capital * 100
                    ann_return   = total_return / (oos_months / 12)
                    ann_vol      = bt_data["strategy_value"].pct_change().std() * np.sqrt(12) * 100
                    summary_data.append({
                        "Risk Aversion":  f"{ra:.2f}",
                        "Final Value":    f"${final_val:,.0f}",
                        "Total Return %": f"{total_return:.2f}%",
                        "Annual Return %":f"{ann_return:.2f}%",
                        "Annual Vol %":   f"{ann_vol:.2f}%",
                    })
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# ============================================================================
# TAB: BETA EXPLORER
# ============================================================================
with tab4:
    st.header("Beta Explorer")
    st.markdown("Understand the factor landscape before committing to target betas.")

    # ── Factor monthly returns ────────────────────────────────────────────────
    st.subheader("📈 Factor Monthly Returns over Lookback Window")
    try:
        ff_plot = fama_french_data.copy()
        ff_plot["Date"] = pd.to_datetime(ff_plot["Date"])
        ff_plot = ff_plot.sort_values("Date").set_index("Date")
        factor_cols_avail = [c for c in ["MF", "SMB", "HML"] if c in ff_plot.columns]

        oos_dt = pd.Period(oos_start, "M").to_timestamp(how="end")
        lb_dt  = oos_dt - pd.DateOffset(months=lookback_months)
        ff_window = ff_plot.loc[lb_dt:oos_dt, factor_cols_avail].copy()
        ff_window.index = ff_window.index.to_period("M")

        colors_map  = {"MF": "#1f77b4", "SMB": "#2ca02c", "HML": "#d62728"}
        factor_full = {"MF": "Market (MF)", "SMB": "Size (SMB)", "HML": "Value (HML)"}

        fig_trend, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for ax, col in zip(axes, factor_cols_avail):
            series = ff_window[col] * 100
            bar_colors = [colors_map[col] if v >= 0 else "#cc0000" for v in series.values]
            ax.bar(range(len(series)), series.values, color=bar_colors, alpha=0.7, width=0.8)
            roll3 = series.rolling(3).mean()
            ax.plot(range(len(series)), roll3.values, color="black", linewidth=1.2, linestyle="--", label="3m avg")
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_ylabel(f"{col} (%)", fontsize=10)
            ax.set_title(factor_full.get(col, col), fontsize=10, fontweight="bold", pad=3)
            ax.grid(True, alpha=0.25, axis="y")
            ax.legend(fontsize=8, loc="upper right")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))

        n = len(ff_window)
        step = max(1, n // 12)
        tick_pos    = list(range(0, n, step))
        tick_labels = [str(ff_window.index[i]) for i in tick_pos]
        axes[-1].set_xticks(tick_pos)
        axes[-1].set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        axes[-1].set_xlabel("Month")
        fig_trend.suptitle(f"Factor Monthly Returns  ({lb_dt.strftime('%Y-%m')} → {oos_dt.strftime('%Y-%m')})",
                           fontsize=12, fontweight="bold")
        fig_trend.tight_layout()
        st.pyplot(fig_trend, use_container_width=True)

        stats_df = pd.DataFrame({
            "Ann. Mean (%)":  (ff_window.mean() * 12).round(2),
            "Ann. Vol (%)":   (ff_window.std() * np.sqrt(12)).round(2),
            "Sharpe":         ((ff_window.mean() / ff_window.std()) * np.sqrt(12)).round(3),
            "Min month (%)":  ff_window.min().round(2),
            "Max month (%)":  ff_window.max().round(2),
        })
        st.dataframe(stats_df, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not render factor trend: {e}")

    st.divider()

    # ── Achievable beta ranges ────────────────────────────────────────────────
    st.subheader("🎯 Achievable Beta Ranges")
    st.caption(f"Min/max portfolio beta reachable given **K_max={K_max}** and **w_max={w_max*100:.0f}%** at **{oos_start}**.")

    if is_recommended:
        st.info("Achievable beta ranges are shown for reference. In Recommended mode the optimizer targets betas from your Excel log automatically.")

    if st.button("Compute Beta Ranges", type="primary"):
        with st.spinner("Estimating betas and computing ranges (MILP)..."):
            try:
                from utils import (estimate_betas_asof_nifty,
                                   compute_achievable_beta_bounds,
                                   _filter_tickers_by_history)

                asof_ts    = pd.Timestamp(oos_start) + pd.offsets.MonthEnd(0)
                tickers_sel = ticker_universe.get(
                    asof_ts,
                    _filter_tickers_by_history(
                        R_full.columns.tolist(), R_full, asof_ts,
                        lookback_months=lookback_months, min_obs=lookback_months,
                    )
                )
                betas_sel = estimate_betas_asof_nifty(
                    returns_df=stock_returns_data, factors_df=fama_french_data,
                    asof=oos_start, tickers_in_window=tickers_sel,
                    lookback_months=lookback_months, min_obs=lookback_months,
                    use_t_as_last_obs=False,
                )
                asof_dt = asof_ts
                lb_dt2  = asof_dt - pd.DateOffset(months=lookback_months)
                R_sel   = R_full.loc[
                    (R_full.index > lb_dt2) & (R_full.index <= asof_dt),
                    [t for t in tickers_sel if t in R_full.columns]
                ].dropna(axis=1, thresh=lookback_months)

                bounds = compute_achievable_beta_bounds(R=R_sel, betas_asof=betas_sel, K_max=K_max, w_max=w_max)

                if "error" in bounds:
                    st.error(f"Cannot compute ranges: {bounds['error']}")
                else:
                    _tgt = target_betas if target_betas else {"MF": 1.0, "SMB": 0.0, "HML": 0.2}
                    ranges_df = pd.DataFrame({
                        f: {
                            "Min achievable": round(v["min"], 3),
                            "Max achievable": round(v["max"], 3),
                            "Your target":    round(_tgt.get(f, 0), 3),
                            "In range?":      "✅" if (v["min"] - 0.01 <= _tgt.get(f, 0) <= v["max"] + 0.01) else "❌",
                        }
                        for f, v in bounds.items()
                    }).T
                    st.dataframe(ranges_df, use_container_width=True)

                    fig_b, ax_b = plt.subplots(figsize=(8, 4))
                    factors_b = list(bounds.keys())
                    mins_b = [bounds[f]["min"] for f in factors_b]
                    maxs_b = [bounds[f]["max"] for f in factors_b]
                    tgts_b = [_tgt.get(f, 0) for f in factors_b]
                    x_b    = np.arange(len(factors_b))
                    ax_b.bar(x_b, [mx - mn for mn, mx in zip(mins_b, maxs_b)], bottom=mins_b,
                             color=["#1f77b4","#2ca02c","#d62728"], alpha=0.4, width=0.5)
                    ax_b.scatter(x_b, tgts_b, color="black", zorder=5, s=100, marker="D", label="Target")
                    ax_b.set_xticks(x_b)
                    ax_b.set_xticklabels(factors_b, fontsize=12)
                    ax_b.set_ylabel("Beta")
                    ax_b.set_title("Achievable Beta Ranges")
                    ax_b.legend()
                    ax_b.grid(True, alpha=0.3, axis="y")
                    st.pyplot(fig_b, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

# ============================================================================
# TAB: INFO
# ============================================================================
with tab5:
    st.header("About This Strategy")
    st.markdown("""
### 📚 Fama-French 3-Factor Model

This application implements a portfolio optimization framework based on the
**Fama-French 3-Factor Model**:

1. **Market Factor (MF)**: Broad market risk exposure
2. **Size Factor (SMB)**: Small-cap vs large-cap premium
3. **Value Factor (HML)**: High book-to-market vs low premium

### 🎯 Strategy Modes

**Customized Strategy**: Full control over target betas, tolerances, sector constraints, etc.

**Recommended Strategy**: Betas are automatically sourced from `beta_search_log.xlsx` (generated by Optuna search in model.ipynb).
- 1. Best betas strategy
- 2. Mean Lookbck period strategy
- 3. Median lookback period strategy
- 4. Monthly mean best betas strategy
- 5. Mean past best betas (6,6,1) strategy
- Strategy 6 mean of regime best betas

### 📊 Key Features

- **Rolling Beta Estimation**: 36-month rolling window OLS regression
- **MILP Optimization**: Mixed-Integer Linear Programming via PuLP/HiGHS solver
- **Turnover Tracking**: Dollar trading volume shown for every rebalance
- **Full-Corpus Universe**: Any ticker with 36+ months history is eligible

### 📁 Data Structure

- `nifty_stocks_data_clean.csv`: Monthly stock returns
- `nifty50_index_data.csv`: Benchmark index returns
- `FF_Nifty50.csv`: Fama-French factors
- `sector_classification.csv`: Ticker-to-sector mapping
- `beta_search_log.xlsx`: Optuna beta search results and optional regime sheets (needed for Recommended Strategy)
    """)

st.divider()
st.markdown("""
<div style='text-align: center'>
    <small>Factor Model Portfolio Optimizer v1.0 | Built with Streamlit</small>
</div>
""", unsafe_allow_html=True)
