import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — avoids GUI memory overhead
import matplotlib.pyplot as plt
import io
from datetime import datetime
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add notebook functions to path if needed
sys.path.insert(0, os.path.dirname(__file__))

# ============================================================================
# MODULE-LEVEL CACHED FUNCTIONS  (must be at top level so cache persists)
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

    # Build figure as PNG bytes so it never needs to be recreated on rerender
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
    fig_bytes = buf.getvalue()

    return sector_monthly, metrics, fig_bytes

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
# SIDEBAR - LOAD DATA AND CONFIGURE PARAMETERS
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data loading - Auto-detect path
    st.subheader("📁 Data Files")
    
    # Auto-detect if running locally or on cloud
    default_data_dir = "data"  # Use data directory
    if os.path.exists("data/nifty_stocks_data (1).csv"):
        data_dir = "data"
        st.info("✅ Using local data files")
    else:
        # Try absolute path (local machine)
        if os.path.exists("c:\\Users\\kshit\\Personal_Factor_model\\data\\nifty_stocks_data (1).csv"):
            data_dir = "c:\\Users\\kshit\\Personal_Factor_model\\data"
            st.info("✅ Using local absolute path")
        else:
            data_dir = st.text_input(
                "Data Directory (if auto-detect failed)",
                value="data",
                help="Path where CSV files are located"
            )
    
    try:
        # Load data with proper path handling
        @st.cache_data
        def load_data(data_dir):
            # Normalize path
            data_dir = os.path.normpath(data_dir)
            
            stock_returns = pd.read_csv(os.path.join(data_dir, 'nifty_stocks_data (1).csv'))
            index_returns = pd.read_csv(os.path.join(data_dir, 'nifty50_index_data.csv'))
            fama_french = pd.read_csv(os.path.join(data_dir, 'FF_Nifty50.csv'))
            yearly_tickers = pd.read_csv(os.path.join(data_dir, 'Nifty_50.csv'))
            sector_df = pd.read_csv(os.path.join(data_dir, 'sector_classification.csv'))
            return stock_returns, index_returns, fama_french, yearly_tickers, sector_df

        stock_returns_data, index_returns_data, fama_french_data, yearly_tickers_data, sector_data = load_data(data_dir)
        ticker_to_sector = dict(zip(sector_data['company'], sector_data['sector']))
        all_sectors = sorted(sector_data['sector'].unique().tolist())
        st.success("✅ Data loaded successfully")
        
    except FileNotFoundError as e:
        st.error(f"""
        ❌ Error loading data: {str(e)}
        
        **Data files not found at**: {data_dir}
        
        **Required files:**
        - nifty_stocks_data (1).csv
        - nifty50_index_data.csv
        - FF_Nifty50.csv
        - Nifty_50.csv
        
        Make sure all CSV files are in the same directory as this app.
        """)
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()
    
    st.divider()
    
    # Backtest Parameters
    st.subheader("📈 Backtest Parameters")
    
    # Generate all months from the data
    stock_returns_data["Date"] = pd.to_datetime(stock_returns_data["Date"])
    min_date = stock_returns_data["Date"].min()
    max_date = stock_returns_data["Date"].max()
    
    # Create list of all month-end dates
    all_months = pd.period_range(start=min_date, end=max_date, freq='M')
    oos_options = [str(m) for m in all_months]
    
    oos_start = st.selectbox(
        "Out-of-Sample Start",
        options=oos_options,
        index=max(0, len(oos_options) - 60),  # Default to ~5 years ago from latest
        help="Start month for backtest"
    )
    
    oos_months = st.slider(
        "OOS Duration (months)",
        min_value=6, max_value=48, value=24, step=1,
        help="Number of months to backtest"
    )
    
    lookback_months = st.slider(
        "Lookback Period (months)",
        min_value=12, max_value=120, value=36, step=12,
        help="Months used for beta estimation"
    )
    
    rebalance_every = st.slider(
        "Rebalance Frequency (months)",
        min_value=1, max_value=12, value=3, step=1,
        help="Rebalance portfolio every N months"
    )
    
    st.divider()
    
    # Optimization Parameters
    st.subheader("🎯 Optimization Constraints")
    
    col1, col2 = st.columns(2)
    with col1:
        K_max = st.slider(
            "Max Positions",
            min_value=5, max_value=50, value=15, step=1,
            help="Maximum number of stocks in portfolio"
        )
    
    with col2:
        w_max = st.slider(
            "Max Position Size",
            min_value=0.05, max_value=0.5, value=0.20, step=0.05,
            help="Maximum weight per stock (as %) × 100"
        )
    
    risk_aversion = st.slider(
        "Risk Aversion",
        min_value=0.1, max_value=10.0, value=1.0, step=0.1,
        help="Higher = more conservative"
    )
    
    st.divider()
    
    # Beta Targets and Tolerances
    st.subheader("📊 Target Betas & Tolerances")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Target Betas**")
        target_mf = st.number_input("Market Beta (MF)", value=1.0, step=0.1)
        target_smb = st.number_input("Size Beta (SMB)", value=0.0, step=0.1)
        target_hml = st.number_input("Value Beta (HML)", value=0.2, step=0.1)
    
    with col2:
        st.write("**Tolerances (±)**")
        tol_mf = st.number_input("MF Tolerance", value=0.3, step=0.05)
        tol_smb = st.number_input("SMB Tolerance", value=0.3, step=0.05)
        tol_hml = st.number_input("HML Tolerance", value=0.3, step=0.05)
    
    target_betas = {"MF": target_mf, "SMB": target_smb, "HML": target_hml}
    beta_tolerances = {"MF": tol_mf, "SMB": tol_smb, "HML": tol_hml}
    
    st.divider()
    
    st.divider()

    # Turnover Cap
    st.subheader("🔁 Turnover Constraint")
    use_turnover_cap = st.checkbox(
        "Enable Turnover Cap",
        value=False,
        help="Limit total portfolio turnover per rebalance"
    )
    if use_turnover_cap:
        turnover_cap = st.slider(
            "Turnover Cap (fraction of portfolio)",
            min_value=0.05, max_value=1.0, value=0.30, step=0.05,
            help="Max sum of |w_new - w_old| per rebalance (e.g. 0.30 = 30% turnover)"
        )
    else:
        turnover_cap = None

    st.divider()

    # Sector Constraints
    st.subheader("🏭 Sector Constraints")
    use_sector_constraints = st.checkbox(
        "Enable Sector Constraints",
        value=False,
        help="Set min/max number of stocks per sector"
    )

    sector_constraints = None
    if use_sector_constraints:
        st.caption("Tick a sector to constrain it. Set max=0 to fully exclude it.")
        sector_constraints = {}
        for sec in all_sectors:
            enabled = st.checkbox(sec, value=False, key=f"sec_enabled_{sec}")
            if enabled:
                col_a, col_b = st.columns(2)
                with col_a:
                    mn = st.number_input("min", min_value=0, max_value=20, value=0, step=1, key=f"sec_min_{sec}")
                with col_b:
                    mx = st.number_input("max", min_value=0, max_value=20, value=5, step=1, key=f"sec_max_{sec}")
                sector_constraints[sec] = (mn if mn > 0 else None, mx)
        if not sector_constraints:
            sector_constraints = None

    st.divider()

    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=10000, max_value=10000000, value=100000, step=10000,
        help="Starting portfolio value"
    )

# ============================================================================
# MAIN TAB INTERFACE
# ============================================================================
tab4, tab6, tab1, tab2, tab3, tab5 = st.tabs(["📐 Beta Explorer", "🏭 Sector Dynamics", "📊 Run Backtest", "📈 Results", "🔍 Risk Analysis", "ℹ️ Info"])

# ============================================================================
# TAB 6: SECTOR DYNAMICS
# ============================================================================
with tab6:
    st.header("Sector Dynamics")

    _sd_key = f"{oos_start}|{lookback_months}"   # track when params change

    # Invalidate stored results if oos_start or lookback changed
    if st.session_state.get("sd_key") != _sd_key:
        st.session_state.pop("sd_results", None)

    if "sd_results" not in st.session_state:
        st.info(f"Configure **As-of date** ({oos_start}) and **Lookback** ({lookback_months}m) in the sidebar, then click below.")
        if st.button("▶️ Compute Sector Dynamics", type="primary", use_container_width=True):
            with st.spinner("Computing sector metrics..."):
                try:
                    sector_monthly, metrics, fig_bytes = compute_sector_dynamics(
                        oos_start, lookback_months,
                        stock_returns_data, fama_french_data, ticker_to_sector
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
        if st.button("🔄 Recompute", help="Recompute after changing sidebar parameters"):
            st.session_state.pop("sd_results", None)
            st.rerun()

        # ── Metrics table ────────────────────────────────────────────────
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

        # ── Trend charts — displayed from pre-rendered PNG bytes ─────────
        st.subheader("📈 Cumulative Return Trends")
        st.caption("Each panel shows equal-weighted cumulative return over the lookback period.")
        st.image(fig_bytes, use_container_width=True)


# ============================================================================
# TAB 1: RUN BACKTEST
# ============================================================================
with tab1:
    st.header("Run Portfolio Backtest")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Backtest Configuration:**
        - Period: {oos_start} for {oos_months} months
        - Lookback: {lookback_months} months
        - Rebalance: Every {rebalance_every} months
        - Portfolio Size: {K_max} stocks, max {w_max*100:.0f}% per position
        - Turnover Cap: {f"{turnover_cap*100:.0f}%" if turnover_cap is not None else "None (unconstrained)"}
        """)
    
    with col2:
        st.info(f"""
        **Target Betas:**
        - Market (MF): {target_mf} ± {tol_mf}
        - Size (SMB): {target_smb} ± {tol_smb}
        - Value (HML): {target_hml} ± {tol_hml}
        """)
    
    if st.button("▶️ Run Backtest", use_container_width=True, type="primary"):
        st.session_state.backtest_running = True
        
        # Import notebook functions - try multiple approaches
        backtest_func = None
        import_error = None
        
        try:
            # Try direct import from utils
            from utils import backtest_fixed_window_quarterly_rebalance_on_breach
            backtest_func = backtest_fixed_window_quarterly_rebalance_on_breach
        except (ImportError, ModuleNotFoundError, SyntaxError) as e:
            import_error = str(e)
            try:
                # Try importing from notebook context (if running in Jupyter kernel)
                from model import backtest_fixed_window_quarterly_rebalance_on_breach
                backtest_func = backtest_fixed_window_quarterly_rebalance_on_breach
            except (ImportError, ModuleNotFoundError, SyntaxError) as e2:
                import_error = str(e2)
        
        if backtest_func is None:
            st.error(f"""
            ❌ Cannot import backtest functions.
            
            **Error Details**: {import_error}
            
            **This is expected on first Streamlit Cloud deployment.**
            
            To fix this:
            
            1. Reload the app page (F5)
            2. Try the backtest again
            
            If error persists, the issue is with the utils.py file. Check:
            - All required dependencies are installed
            - No syntax errors in utils.py
            - File is in the correct directory
            
            See DEPLOYMENT_GUIDE.md for detailed troubleshooting.
            """)
            st.stop()
        
        with st.spinner(f"⏳ Running backtest from {oos_start}..."):
            try:
                backtest_result = backtest_func(
                    stock_returns_data=stock_returns_data,
                    fama_french_data=fama_french_data,
                    index_returns=index_returns_data,
                    universe_by_year=yearly_tickers_data,
                    
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
                    show_progress=True
                )
                
                st.session_state.backtest_result = backtest_result
                st.success("✅ Backtest completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Backtest failed: {str(e)}")
                st.exception(e)

# ============================================================================
# TAB 2: RESULTS
# ============================================================================
with tab2:
    st.header("Backtest Results")
    
    if "backtest_result" not in st.session_state:
        st.info("👈 Run a backtest first in the 'Run Backtest' tab")
    else:
        bt = st.session_state.backtest_result
        strategy_value = bt["strategy_value"]
        index_value = bt["index_value"]
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        final_strategy = strategy_value.iloc[-1]
        final_index = index_value.iloc[-1]
        strategy_return = ((final_strategy - initial_capital) / initial_capital) * 100
        index_return = ((final_index - initial_capital) / initial_capital) * 100
        
        with col1:
            st.metric(
                "Customized Portfolio Final Value",
                f"${final_strategy:,.0f}",
                f"{strategy_return:+.2f}%"
            )
        
        with col2:
            st.metric(
                "Nifty 50 Final Value",
                f"${final_index:,.0f}",
                f"{index_return:+.2f}%"
            )
        
        with col3:
            outperformance = ((final_strategy - final_index) / final_index) * 100
            st.metric(
                "Outperformance vs Nifty 50",
                f"{outperformance:+.2f}%",
                f"${final_strategy - final_index:+,.0f}"
            )
        
        with col4:
            annual_vol_strategy = strategy_value.pct_change().std() * np.sqrt(12)
            st.metric(
                "Annual Volatility",
                f"{annual_vol_strategy:.2%}",
                "Customized Portfolio"
            )
        
        st.divider()
        
        # Portfolio value chart
        st.subheader("📈 Portfolio Value Over Time")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(strategy_value.index, strategy_value.values, label="Customized Portfolio", linewidth=2, color="#1f77b4")
        ax.plot(index_value.index, index_value.values, label="Nifty 50", linewidth=2, color="#ff7f0e")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_title("Strategy vs Index Performance")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        
        # Returns comparison
        st.subheader("📊 Monthly Returns")
        
        strategy_monthly = strategy_value.pct_change()
        index_monthly = index_value.pct_change()
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(len(strategy_monthly)), strategy_monthly.values, label="Customized Portfolio", alpha=0.7, color="#1f77b4")
        ax.bar(range(len(index_monthly)), index_monthly.values, label="Nifty 50", alpha=0.7, color="#ff7f0e")
        ax.set_xlabel("Month")
        ax.set_ylabel("Return")
        ax.set_title("Monthly Returns Comparison")
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        
        # Rebalancing log
        if "rebalance_log" in bt and bt["rebalance_log"]:
            st.subheader("🔄 Rebalancing Log")
            
            rebalance_log = bt["rebalance_log"]
            log_df = pd.DataFrame(rebalance_log)
            
            st.dataframe(
                log_df.style.format({
                    'exposures': lambda x: str({k: f"{v:.3f}" for k, v in x.items()})
                }),
                use_container_width=True
            )
            
            st.divider()
            
            st.subheader("📊 Portfolio Composition at Each Quarter")
            
            # Extract portfolio compositions from rebalance log
            compositions = {}
            for date_str, log_entry in rebalance_log.items():
                weights = log_entry.get("weights", {})
                asof_date = log_entry.get("asof", pd.to_datetime(date_str))
                
                # Convert weights to dict if it's a Series
                if isinstance(weights, pd.Series):
                    weights = weights.to_dict()
                
                if weights:  # Only process if weights is not empty
                    # Create quarter label (e.g., "2022-Q4") or use date format (e.g., "2022/09")
                    if isinstance(asof_date, str):
                        asof_date = pd.to_datetime(asof_date)
                    quarter_label = f"{asof_date.year}/{asof_date.month:02d}"
                    compositions[quarter_label] = weights
            
            if compositions:
                # Build composition dataframe
                comp_df = pd.DataFrame.from_dict(compositions, orient='index')
                comp_df = comp_df.fillna(0.0)
                
                # Convert to percentages
                comp_df_pct = (comp_df * 100).round(2)
                
                # Display table with better formatting
                st.write("**Weights (%) at each rebalancing date:**")
                st.dataframe(
                    comp_df_pct.style
                        .format("{:.2f}")
                        .highlight_max(axis=1, color='#FFE5B4')
                        .highlight_min(axis=1, color='#E8F5E9'),
                    use_container_width=True,
                    height=500
                )
                
                # Show summary: top stocks overall
                st.write("**Average position size across all periods:**")
                avg_weights = comp_df.mean().sort_values(ascending=False) * 100
                avg_weights = avg_weights[avg_weights > 0.1]  # Only show stocks with avg weight > 0.1%
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(
                        avg_weights.to_frame("Avg Weight (%)").style.format("{:.2f}"),
                        use_container_width=True
                    )
                
                with col2:
                    # Pie chart of average allocation
                    if len(avg_weights) > 0:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.pie(avg_weights, labels=avg_weights.index, autopct='%1.1f%%', startangle=90)
                        ax.set_title("Average Portfolio Allocation")
                        st.pyplot(fig, use_container_width=True)
            else:
                st.info("⚠️ No rebalancing occurred during this backtest period")

# ============================================================================
# TAB 3: RISK ANALYSIS
# ============================================================================
with tab3:
    st.header("Risk Sensitivity Analysis")
    
    st.info("Run multiple backtests with varying risk aversion parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_ra = st.slider("Min Risk Aversion", 0.1, 5.0, 1.0, step=0.1)
    
    with col2:
        max_ra = st.slider("Max Risk Aversion", 1.0, 10.0, 5.0, step=0.1)
    
    num_scenarios = st.slider("Number of Scenarios", 2, 20, 5, step=1)
    
    if st.button("🔄 Run Risk Sensitivity Analysis", use_container_width=True, type="primary"):
        backtest_func = None
        import_error = None
        
        try:
            from utils import backtest_fixed_window_quarterly_rebalance_on_breach
            backtest_func = backtest_fixed_window_quarterly_rebalance_on_breach
        except (ImportError, ModuleNotFoundError, SyntaxError) as e:
            import_error = str(e)
            try:
                from model import backtest_fixed_window_quarterly_rebalance_on_breach
                backtest_func = backtest_fixed_window_quarterly_rebalance_on_breach
            except (ImportError, ModuleNotFoundError, SyntaxError) as e2:
                import_error = str(e2)
        
        if backtest_func is None:
            st.error(f"Cannot import backtest functions. Error: {import_error}")
            st.stop()
        
        risk_aversion_values = np.linspace(min_ra, max_ra, num_scenarios)
        sensitivity_results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ra in enumerate(risk_aversion_values):
            status_text.text(f"Running scenario {i+1}/{num_scenarios} (Risk Aversion={ra:.2f})...")
            
            try:
                bt_temp = backtest_func(
                    stock_returns_data=stock_returns_data,
                    fama_french_data=fama_french_data,
                    index_returns=index_returns_data,
                    universe_by_year=yearly_tickers_data,
                    
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
                    show_progress=False
                )
                
                sensitivity_results[ra] = bt_temp
                
            except Exception as e:
                st.warning(f"Scenario with RA={ra:.2f} failed: {str(e)}")
            
            progress_bar.progress((i + 1) / num_scenarios)
        
        status_text.empty()
        
        if sensitivity_results:
            st.session_state.sensitivity_results = sensitivity_results
            st.success(f"✅ Completed {len(sensitivity_results)} scenarios")
            
            # Plot all scenarios
            st.subheader("📈 Risk Aversion Sensitivity")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for ra, bt_data in sorted(sensitivity_results.items()):
                ax.plot(
                    bt_data["strategy_value"].index,
                    bt_data["strategy_value"].values,
                    label=f"RA={ra:.2f}",
                    linewidth=2,
                    alpha=0.8
                )
            
            ax.set_xlabel("Date")
            ax.set_ylabel("Portfolio Value ($)")
            ax.set_title("Customized Portfolio Performance Across Risk Aversion Levels")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            
            # Summary table
            st.subheader("📊 Summary Statistics")
            
            summary_data = []
            for ra, bt_data in sorted(sensitivity_results.items()):
                final_val = bt_data["strategy_value"].iloc[-1]
                total_return = ((final_val - initial_capital) / initial_capital) * 100
                annual_return = total_return / (oos_months / 12)
                annual_vol = bt_data["strategy_value"].pct_change().std() * np.sqrt(12) * 100
                
                summary_data.append({
                    "Risk Aversion": f"{ra:.2f}",
                    "Final Value": f"${final_val:,.0f}",
                    "Total Return %": f"{total_return:.2f}%",
                    "Annual Return %": f"{annual_return:.2f}%",
                    "Annual Vol %": f"{annual_vol:.2f}%"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

# ============================================================================
# TAB 4: BETA EXPLORER
# ============================================================================
with tab4:
    st.header("Beta Explorer")
    st.markdown("Understand the factor landscape **before** committing to target betas.")

    # ── Section 1: Factor Monthly Values (3 separate windows) ───────────────
    st.subheader("📈 Factor Monthly Returns over Lookback Window")
    st.caption("Each panel shows the actual monthly factor value over the selected lookback period.")

    try:
        ff_plot = fama_french_data.copy()
        ff_plot["Date"] = pd.to_datetime(ff_plot["Date"])
        ff_plot = ff_plot.sort_values("Date").set_index("Date")

        factor_cols_avail = [c for c in ["MF", "SMB", "HML"] if c in ff_plot.columns]

        oos_dt = pd.Period(oos_start, "M").to_timestamp(how="end")
        lb_dt  = oos_dt - pd.DateOffset(months=lookback_months)
        ff_window = ff_plot.loc[lb_dt:oos_dt, factor_cols_avail].copy()
        ff_window.index = ff_window.index.to_period("M")  # month labels on x-axis

        colors_map  = {"MF": "#1f77b4", "SMB": "#2ca02c", "HML": "#d62728"}
        factor_full = {"MF": "Market (MF)", "SMB": "Size (SMB)", "HML": "Value (HML)"}

        fig_trend, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        for ax, col in zip(axes, factor_cols_avail):
            series = ff_window[col] * 100          # convert to %
            color  = colors_map[col]

            # colour bars green/red by sign
            bar_colors = [color if v >= 0 else "#cc0000" for v in series.values]
            ax.bar(range(len(series)), series.values, color=bar_colors, alpha=0.7, width=0.8)

            # rolling 3-month average line
            roll3 = series.rolling(3).mean()
            ax.plot(range(len(series)), roll3.values,
                    color="black", linewidth=1.2, linestyle="--", label="3m avg")

            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_ylabel(f"{col} (%)", fontsize=10)
            ax.set_title(factor_full.get(col, col), fontsize=10, fontweight="bold", pad=3)
            ax.grid(True, alpha=0.25, axis="y")
            ax.legend(fontsize=8, loc="upper right")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))

        # x-axis: show every ~6th month label to avoid crowding
        n = len(ff_window)
        step = max(1, n // 12)
        tick_pos    = list(range(0, n, step))
        tick_labels = [str(ff_window.index[i]) for i in tick_pos]
        axes[-1].set_xticks(tick_pos)
        axes[-1].set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        axes[-1].set_xlabel("Month")

        fig_trend.suptitle(
            f"Factor Monthly Returns  ({lb_dt.strftime('%Y-%m')} → {oos_dt.strftime('%Y-%m')})",
            fontsize=12, fontweight="bold"
        )
        fig_trend.tight_layout()
        st.pyplot(fig_trend, use_container_width=True)

        # Quick stats table
        st.caption("**Factor statistics over selected lookback window:**")
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

    # ── Section 2: Achievable Beta Ranges ────────────────────────────────────
    st.subheader("🎯 Achievable Beta Ranges")
    st.caption(f"Min/max portfolio beta reachable given **K_max={K_max}** and **w_max={w_max*100:.0f}%** at **{oos_start}**.")
    st.info("This tells you which target betas are actually feasible before you run the backtest.")

    if st.button("Compute Beta Ranges", type="primary"):
        with st.spinner("Estimating betas and computing ranges (MILP)..."):
            try:
                from utils import (estimate_betas_asof_nifty,
                                   compute_achievable_beta_bounds)

                year_sel = pd.Period(oos_start, "M").to_timestamp().year
                tickers_sel = (
                    yearly_tickers_data[yearly_tickers_data["Year"] == year_sel]
                    .drop(columns=["Year"]).iloc[0].dropna().tolist()
                )

                betas_sel = estimate_betas_asof_nifty(
                    returns_df       = stock_returns_data,
                    factors_df       = fama_french_data,
                    asof             = oos_start,
                    tickers_in_window= tickers_sel,
                    lookback_months  = lookback_months,
                    min_obs          = 24,
                    use_t_as_last_obs= False,
                )

                # Build R from stock returns
                sr = stock_returns_data.copy()
                sr["Date"] = pd.to_datetime(sr["Date"])
                asof_dt  = pd.Period(oos_start, "M").to_timestamp(how="end")
                lb_dt2   = asof_dt - pd.DateOffset(months=lookback_months)
                sr_win   = sr[(sr["Date"] > lb_dt2) & (sr["Date"] <= asof_dt)]
                R_sel    = sr_win.pivot(index="Date", columns="Ticker", values="RET")
                R_sel    = R_sel[R_sel.columns.intersection(tickers_sel)].dropna(axis=1, thresh=24)

                bounds = compute_achievable_beta_bounds(
                    R          = R_sel,
                    betas_asof = betas_sel,
                    K_max      = K_max,
                    w_max      = w_max,
                )

                if "error" in bounds:
                    st.error(f"Cannot compute ranges: {bounds['error']}")
                else:
                    # Table
                    ranges_df = pd.DataFrame({
                        f: {"Min achievable": round(v["min"], 3),
                            "Max achievable": round(v["max"], 3),
                            "Your target":    round(target_betas.get(f, 0), 3),
                            "In range?":      "✅" if (v["min"] - 0.01
                                                       <= target_betas.get(f, 0)
                                                       <= v["max"] + 0.01) else "❌"}
                        for f, v in bounds.items()
                    }).T
                    st.dataframe(ranges_df, use_container_width=True)

                    # Bar chart
                    fig_b, ax_b = plt.subplots(figsize=(8, 4))
                    factors_b = list(bounds.keys())
                    mins_b  = [bounds[f]["min"] for f in factors_b]
                    maxs_b  = [bounds[f]["max"] for f in factors_b]
                    tgts_b  = [target_betas.get(f, 0) for f in factors_b]
                    x_b     = np.arange(len(factors_b))
                    bar_h   = [mx - mn for mn, mx in zip(mins_b, maxs_b)]

                    ax_b.bar(x_b, bar_h, bottom=mins_b, color=["#1f77b4","#2ca02c","#d62728"],
                             alpha=0.4, width=0.5, label="Achievable range")
                    ax_b.scatter(x_b, tgts_b, color="black", zorder=5,
                                 s=100, marker="D", label="Your target")

                    for xi, mn, mx, tg in zip(x_b, mins_b, maxs_b, tgts_b):
                        ax_b.text(xi, mn - 0.05, f"{mn:.2f}", ha="center", fontsize=9, color="grey")
                        ax_b.text(xi, mx + 0.03, f"{mx:.2f}", ha="center", fontsize=9, color="grey")
                        ax_b.text(xi, tg + 0.03,  f"↑{tg:.2f}", ha="center", fontsize=9,
                                  color="black", fontweight="bold")

                    ax_b.set_xticks(x_b)
                    ax_b.set_xticklabels(factors_b, fontsize=12)
                    ax_b.set_ylabel("Beta")
                    ax_b.set_title("Achievable Beta Ranges vs Your Targets")
                    ax_b.legend()
                    ax_b.grid(True, alpha=0.3, axis="y")
                    st.pyplot(fig_b, use_container_width=True)

            except Exception as e:
                st.error(f"Error computing beta ranges: {e}")
                st.exception(e)

# ============================================================================
# TAB 5: INFO
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
    
    ### 🎯 Optimization Approach
    
    - **Objective**: Maximize expected return while penalizing risk (Mean Absolute Deviation)
    - **Constraints**:
      - Fully invested portfolio
      - Target factor exposures with user-defined tolerances
      - Cardinality constraint (max positions)
      - Position size limits
    
    ### 📊 Key Features
    
    - **Rolling Beta Estimation**: 36-60 month rolling window OLS regression
    - **Quarterly Rebalancing**: Breach-triggered rebalancing when factor exposures drift
    - **MILP Optimization**: Mixed-Integer Linear Programming via PuLP/CBC solver
    - **Sensitivity Analysis**: Test multiple risk aversion parameters
    
    ### 📁 Data Structure
    
    Required CSV files in data directory:
    - `nifty_stocks_data (1).csv`: Daily/monthly stock returns
    - `nifty50_index_data.csv`: Benchmark index returns
    - `FF_Nifty50.csv`: Fama-French factors (MF, SMB, HML, RF)
    - `Nifty_50.csv`: Available tickers by year
    
    ### ⚠️ Interpretation Guide
    
    **When optimization is infeasible:**
    - The debug output shows achievable beta ranges
    - Adjust target betas to fall within those ranges
    - Increase tolerances if constraints are too tight
    """)

st.divider()
st.markdown("""
<div style='text-align: center'>
    <small>Factor Model Portfolio Optimizer v1.0 | Built with Streamlit</small>
</div>
""", unsafe_allow_html=True)
