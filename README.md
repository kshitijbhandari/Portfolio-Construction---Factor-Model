# 🚀 Streamlit App - Factor Model Portfolio Optimizer

A web-based interface for running factor model portfolio optimization backtests using Fama-French 3-factors.

## ✨ Features

- **🎯 Interactive Parameter Configuration**: Adjust backtest settings via sidebar controls
- **📊 Portfolio Optimization**: MILP-based optimization with target betas and cardinality constraints  
- **📈 Backtest Execution**: Run backtests from 2020-2023 with quarterly rebalancing
- **🔍 Results Visualization**: Charts, metrics, and rebalancing logs
- **📉 Sensitivity Analysis**: Compare multiple risk aversion scenarios
- **⚠️ Smart Error Messages**: Shows achievable beta ranges when optimization fails

---

## 🚀 Quick Start (3 Steps)

### 1️⃣ Install Dependencies
```bash
cd c:\Users\kshit\Personal_Factor_model
pip install -r requirements.txt
```

### 2️⃣ Setup Functions (Auto or Manual)

**Option A: Auto-Setup (Easiest)**
```bash
python setup_streamlit.py
```

**Option B: Manual Setup**
See [EXTRACT_FUNCTIONS.md](EXTRACT_FUNCTIONS.md) for detailed instructions

### 3️⃣ Run the App
```bash
streamlit run app.py
```

App opens at: **http://localhost:8501**

---

## 📋 File Structure

```
c:\Users\kshit\Personal_Factor_model\
├── app.py                      # Main Streamlit app
├── utils.py                    # Extracted functions from notebook
├── setup_streamlit.py          # Auto-setup script
├── requirements.txt            # Python dependencies
├── model.ipynb                 # Original Jupyter notebook
├── README.md                   # This file
├── DEPLOYMENT_GUIDE.md         # Production deployment
├── EXTRACT_FUNCTIONS.md        # Function extraction guide
│
├── nifty_stocks_data (1).csv   # Stock returns data
├── nifty50_index_data.csv      # Index returns data  
├── FF_Nifty50.csv              # Fama-French factors
└── Nifty_50.csv                # Universe tickers by year
```

---

## 🎮 How to Use

### Tab 1: Run Backtest

1. **Configure in Sidebar**:
   - Select OOS start period (2020-01, 2021-01, etc.)
   - Set duration (months)
   - Adjust optimization parameters (max positions, position size)
   - Set target betas and tolerances

2. **Click "▶️ Run Backtest"**

3. **View Results**:
   - Strategy vs Index comparison
   - Final portfolio value
   - Rebalancing log

**Tip**: If optimization is infeasible, error shows achievable beta ranges:
```
[ACHIEVABLE BOUNDS]
MF: [0.8340, 3.4976]
SMB: [-2.1223, -0.1006]  ← Adjust target to ~-0.5
HML: [-0.5123, 1.3492]
```

### Tab 2: Results

- **Key Metrics**: Final value, returns, outperformance
- **Charts**: Portfolio value over time, monthly returns
- **Rebalancing Log**: All portfolio adjustments with factor exposures

### Tab 3: Risk Analysis

- Run 5-20 backtests with different risk aversion parameters
- Compare performance across scenarios
- Summary table with statistics

### Tab 4: Info

- Model documentation
- Factor explanations
- Data structure guide

---

## ⚙️ Configuration Guide

### Backtest Parameters

| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| OOS Start | 2020-2023 | 2020-01 | Backtest start month |
| OOS Duration | 6-48 months | 24 | Length of backtest |  
| Lookback | 12-120 months | 36 | Months for beta estimation |
| Rebalance Freq | 1-12 months | 3 | Portfolio rebalance frequency |

### Optimization Parameters

| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| Max Positions | 5-50 | 15 | Maximum stocks in portfolio |
| Max Position | 5%-50% | 20% | Max weight per stock |
| Risk Aversion | 0.1-10 | 1.0 | Higher = more conservative |

### Target Betas & Tolerances

| Factor | Default Target | Default Tol | Notes |
|--------|-----------------|-------------|-------|
| MF (Market) | 1.0 | ±0.3 | Market exposure |
| SMB (Size) | 0.0 | ±0.3 | Small-cap exposure |
| HML (Value) | 0.2 | ±0.3 | Value stock exposure |

**Tip**: Check the debug output if optimization fails - it shows which ranges are actually achievable!

---

## 🔧 Troubleshooting

### ❌ "Cannot import backtest functions"
```bash
# Option 1: Run auto-setup
python setup_streamlit.py

# Option 2: Manual extraction
jupyter nbconvert --to script model.ipynb --output utils.py
# Then edit utils.py to keep only function definitions
```

### ❌ "Optimization Failed: Infeasible"
The app shows achievable beta ranges. Adjust:
- **Target betas** to be within achievable ranges
- **Tolerances** to be wider (increase ±)
- **Constraints** like K_max or w_max to be looser

Example fix:
```python
# Before (infeasible)
target_betas={"MF": 1.0, "SMB": 0.0, "HML": 0.2}
beta_tolerances={"MF": 0.1, "SMB": 0.1, "HML": 0.1}

# After (feasible)
target_betas={"MF": 1.0, "SMB": -0.5, "HML": 0.2}
beta_tolerances={"MF": 0.3, "SMB": 0.5, "HML": 0.3}
```

### ❌ "Data not found"
Ensure CSV files are in same directory as app.py:
- ✅ nifty_stocks_data (1).csv
- ✅ nifty50_index_data.csv
- ✅ FF_Nifty50.csv
- ✅ Nifty_50.csv

### ⚠️ Slow Performance
- Reduce OOS duration (12 months instead of 24)
- Increase rebalance frequency (6 months instead of 3)
- Reduce K_max (10 instead of 15)

---

## 📊 Running Multiple Scenarios

### Example Workflow

1. **Run with Conservative Settings**:
   - Risk Aversion: 5.0
   - Target: MF=1.0, SMB=-0.5, HML=0.2
   - Tolerances: 0.3 each

2. **Switch to Aggressive**:
   - Risk Aversion: 1.0
   - Same targets but wider tolerances (0.5)

3. **Compare in Risk Analysis Tab**:
   - Run 10 scenarios from RA 1-5
   - View performance curves
   - Identify optimal risk level

---

## 🌐 Deployment Options

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for:
- **Streamlit Cloud** (Easiest, Free)
- **Heroku** (Paid)
- **AWS EC2** (Scalable)
- **Docker** (Production)

Quick Streamlit Cloud deployment:
```bash
# 1. Push to GitHub
git push origin main

# 2. Go to https://streamlit.io/cloud
# 3. Connect repo and deploy
```

---

## 📚 Understanding the Output

### Optimization Debug Info
When optimization fails, shows:
```
[REQUESTED BETA BANDS]
MF: 1.0000 ± 0.1000 => [0.9000, 1.1000]

[ACHIEVABLE BOUNDS]  
MF: [0.8340, 3.4976]        ← What's actually possible
SMB: [-2.1223, -0.1006]     ← Adjust target to ~-0.5
HML: [-0.5123, 1.3492]

[CONDITIONAL BOUNDS]
SMB: [-1.0886, -0.2718]     ← Range when MF & HML satisfied
```

### Performance Metrics
- **Final Value**: Portfolio value at end of backtest
- **Total Return %**: (Final - Initial) / Initial × 100
- **Annual Return %**: Total Return / (months/12)
- **Annual Vol %**: Monthly return std × √12 × 100

---

## 🎓 Model Documentation

### Fama-French 3-Factor Model

Returns decomposed as:
$$R_i - R_f = \alpha + \beta_{MF}(R_m - R_f) + \beta_{SMB} \cdot SMB + \beta_{HML} \cdot HML + \epsilon$$

Where:
- **MF**: Market factor (broad market excess return)
- **SMB**: Size factor (small-cap minus large-cap return)
- **HML**: Value factor (high B/M minus low B/M return)

### Optimization Problem

Maximize:
$$\mu^T w - \lambda \cdot MAD(R_p)$$

Subject to:
- $\sum w_i = 1$ (fully invested)
- $w_i \leq w_{max} \cdot z_i$ (position limits)
- $\sum z_i \leq K_{max}$ (cardinality)
- $\beta_{target} - \text{tol} \leq B w \leq \beta_{target} + \text{tol}$ (target betas)

Where:
- λ = risk aversion parameter
- MAD = Mean Absolute Deviation
- B = beta matrix

---

## 📞 Support & Resources

**If you get stuck:**

1. **Check error message** - shows achievable beta ranges
2. **Read EXTRACT_FUNCTIONS.md** - for setup issues
3. **Review DEPLOYMENT_GUIDE.md** - for deployment
4. **Adjust constraints** - make beta bands wider
5. **Reduce backtest period** - for faster testing

**External Resources:**
- [Streamlit Docs](https://docs.streamlit.io)
- [PuLP Optimization](https://coin-or.github.io/pulp/)
- [Fama-French Data](https://ken.french/Data_Library.html)

---

## 📝 License & Credits

Built with:
- **Streamlit** - Web framework
- **PuLP** - MILP optimization  
- **Pandas** - Data manipulation
- **Matplotlib** - Visualization
- **NumPy** - Numerical computing

---

**Last Updated**: March 2026  
**Version**: 1.0
