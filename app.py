import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add notebook functions to path if needed
sys.path.insert(0, os.path.dirname(__file__))

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
    default_data_dir = "."  # Use current directory
    if os.path.exists("nifty_stocks_data (1).csv"):
        data_dir = "."
        st.info("✅ Using local data files")
    else:
        # Try absolute path (local machine)
        if os.path.exists("c:\\Users\\kshit\\Personal_Factor_model\\nifty_stocks_data (1).csv"):
            data_dir = "c:\\Users\\kshit\\Personal_Factor_model"
            st.info("✅ Using local absolute path")
        else:
            data_dir = st.text_input(
                "Data Directory (if auto-detect failed)",
                value=".",
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
            return stock_returns, index_returns, fama_french, yearly_tickers
        
        stock_returns_data, index_returns_data, fama_french_data, yearly_tickers_data = load_data(data_dir)
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
    
    oos_start = st.selectbox(
        "Out-of-Sample Start",
        options=["2020-01", "2021-01", "2022-01", "2023-01"],
        index=0,
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
    
    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=10000, max_value=10000000, value=100000, step=10000,
        help="Starting portfolio value"
    )

# ============================================================================
# MAIN TAB INTERFACE
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Run Backtest", "📈 Results", "🔍 Risk Analysis", "ℹ️ Info"])

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
                    turnover_cap=None,
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
                "Strategy Final Value",
                f"${final_strategy:,.0f}",
                f"{strategy_return:+.2f}%"
            )
        
        with col2:
            st.metric(
                "Index Final Value",
                f"${final_index:,.0f}",
                f"{index_return:+.2f}%"
            )
        
        with col3:
            outperformance = ((final_strategy - final_index) / final_index) * 100
            st.metric(
                "Outperformance",
                f"{outperformance:+.2f}%",
                f"${final_strategy - final_index:+,.0f}"
            )
        
        with col4:
            annual_vol_strategy = strategy_value.pct_change().std() * np.sqrt(12)
            st.metric(
                "Annual Volatility",
                f"{annual_vol_strategy:.2%}",
                "Strategy"
            )
        
        st.divider()
        
        # Portfolio value chart
        st.subheader("📈 Portfolio Value Over Time")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(strategy_value.index, strategy_value.values, label="Strategy", linewidth=2, color="#1f77b4")
        ax.plot(index_value.index, index_value.values, label="Index", linewidth=2, color="#ff7f0e")
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
        ax.bar(range(len(strategy_monthly)), strategy_monthly.values, label="Strategy", alpha=0.7, color="#1f77b4")
        ax.bar(range(len(index_monthly)), index_monthly.values, label="Index", alpha=0.7, color="#ff7f0e")
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
                    turnover_cap=None,
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
            ax.set_title("Strategy Performance Across Risk Aversion Levels")
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
# TAB 4: INFO
# ============================================================================
with tab4:
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
