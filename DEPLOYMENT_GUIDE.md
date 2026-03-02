# 🚀 Streamlit Deployment Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Locally
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📋 Setup Instructions

### Prerequisites
- Python 3.8+
- Jupyter Notebook (for development)
- All CSV data files in the same directory as `app.py`:
  - `nifty_stocks_data (1).csv`
  - `nifty50_index_data.csv`
  - `FF_Nifty50.csv`
  - `Nifty_50.csv`

### Development Setup (Windows)
```bash
# Navigate to project directory
cd c:\Users\kshit\Personal_Factor_model

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🌐 Deployment Options

### Option A: Streamlit Cloud (Recommended - Free)
1. Push your project to GitHub
2. Go to https://streamlit.io/cloud
3. Click "New app"
4. Connect your GitHub repo
5. Specify:
   - Repository: `your-username/factor-model`
   - Branch: `main`
   - Main file path: `app.py`
6. Click Deploy

**Note**: Update `data_dir` in app.py to use GitHub URL or upload data to Streamlit secrets.

### Option B: Heroku (Paid)
1. Create `Procfile`:
   ```
   web: streamlit run app.py --logger.level=error --client.showErrorDetails=false
   ```

2. Create `.streamlit/config.toml`:
   ```toml
   [server]
   port = $PORT
   enableCORS = false
   
   [logger]
   level = "error"
   ```

3. Deploy:
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

### Option C: AWS EC2
1. Launch Ubuntu EC2 instance
2. Install Python and dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install python3-pip
   pip install -r requirements.txt
   ```
3. Run in background:
   ```bash
   nohup streamlit run app.py --server.port 8501 &
   ```
4. Access via: `http://your-ec2-ip:8501`

### Option D: Docker (Production)
1. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.10-slim
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py"]
   ```

2. Build and run:
   ```bash
   docker build -t factor-model .
   docker run -p 8501:8501 factor-model
   ```

---

## 📊 App Features

### 1. **Run Backtest Tab**
- Configure backtest parameters
- Adjust optimization constraints
- Set target betas and tolerances
- View infeasibility warnings with achievable beta ranges

### 2. **Results Tab**
- Portfolio performance metrics
- Strategy vs index comparison chart
- Monthly returns breakdown
- Rebalancing log with exposures

### 3. **Risk Analysis Tab**
- Sensitivity analysis across risk aversion parameters
- Multi-scenario performance comparison
- Summary statistics table

### 4. **Info Tab**
- Model documentation
- Factor explanations
- Data structure guide

---

## ⚙️ Configuration

### Key Parameters in Sidebar

**Backtest Settings:**
- `Out-of-Sample Start`: Start month (2020-01, 2021-01, etc.)
- `OOS Duration`: Number of months to backtest (6-48)
- `Lookback Period`: Months for beta estimation (12-120)
- `Rebalance Frequency`: Rebalance every N months (1-12)

**Optimization Constraints:**
- `Max Positions`: Maximum stocks in portfolio (5-50)
- `Max Position Size`: Max weight per stock (5%-50%)
- `Risk Aversion`: Controls return vs risk trade-off (0.1-10.0)

**Target Betas:**
- `Market (MF)`: Target market beta (default: 1.0)
- `Size (SMB)`: Target size beta (default: 0.0)
- `Value (HML)`: Target value beta (default: 0.2)

**Tolerances:**
- `MF/SMB/HML Tolerance`: Allow ± deviation from targets

---

## 🔧 Troubleshooting

### Issue: "Optimization Failed: Infeasible"
**Solution**: Check the error message for achievable beta bounds:
```
[CONDITIONAL BOUNDS]
MF: [-0.5, 1.2]      ← Your target must be in this range
SMB: [-1.1, -0.3]    ← Adjust target_smb to ~-0.7
HML: [0.0, 0.8]      ← Adjust tolerances wider
```

### Issue: "Data not found"
**Solution**: Ensure all CSV files are in the correct directory and paths in sidebar match exactly.

### Issue: "Slow backtest performance"
**Solution**: 
- Reduce `OOS Duration` or `lookback_months`
- Increase `rebalance_every` to 6 or 12 months
- Use smaller `K_max` values

### Issue: "PuLP solver not found"
**Solution**:
```bash
pip install pulp
# For better solver: pip install coinor-cbc
```

---

## 📈 Usage Example

1. **Open the app**: Run `streamlit run app.py`
2. **Configure sidebar**:
   - OOS Start: 2023-01
   - Duration: 12 months
   - Max Positions: 10
   - Risk Aversion: 2.0
3. **Set targets**: MF=1.0±0.2, SMB=-0.5±0.3, HML=0.2±0.2
4. **Click "Run Backtest"**
5. **View results** in Results tab
6. **Run sensitivity** in Risk Analysis tab comparing RA 1-5

---

## 📤 Tips for Production

1. **Data Management**:
   - Store CSV files separately
   - Use environment variables for paths
   - Consider database for larger datasets

2. **Performance**:
   - Add caching: `@st.cache_data`
   - Pre-compute betas for common periods
   - Use multiprocessing for scenarios

3. **Security**:
   - Use Streamlit secrets for API keys
   - Validate all user inputs
   - Set max file upload sizes

4. **Monitoring**:
   - Add logging to track backtest runs
   - Monitor solver performance
   - Alert on infeasible scenarios

---

## 📚 Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **PuLP Docs**: https://coin-or.github.io/pulp/
- **Fama-French Factors**: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- **Streamlit Cloud**: https://streamlit.io/cloud

---

## 🆘 Support

If you encounter issues:
1. Check the error message in the app
2. Review the achievable beta bounds when optimization fails
3. Adjust constraints if infeasible
4. Check CSV file formats match expected structure

---

**Last Updated**: March 2, 2026
