# 📦 How to Extract Functions from Notebook for Streamlit

## Problem
The Streamlit app needs access to Python functions from your Jupyter notebook (model.ipynb). There are two ways to solve this:

---

## Solution 1: Auto-Export Notebook Functions (Easiest)

### Method A: Using a Script
Create a file called `export_functions.py`:

```python
import subprocess
import json

# Convert notebook to Python script
subprocess.run([
    "jupyter", "nbconvert", "--to", "script", 
    "model.ipynb", "--output", "utils.py"
], check=True)

print("✅ Exported model.ipynb to utils.py")
```

Then run:
```bash
python export_functions.py
```

This creates `utils.py` with all functions from the notebook.

### Method B: Manual Copy
1. Open your Jupyter notebook in an IDE (VSCode, PyCharm)
2. Find all cells that define functions (not ones that run backtests)
3. Copy these function definitions:
   - `estimate_betas_asof_nifty()` (Cell 3)
   - `compute_sigma_f_and_lambda()` (Cell 6)
   - `build_mu_and_sigma_from_betas()` (Cell 7)
   - `optimize_pulp_mad_targetbetas_cardinality()` (Cell 8)
   - `compute_achievable_beta_bounds()` (Cell 9)
   - `compute_conditional_beta_bounds()` (Cell 10)
   - `backtest_fixed_window_quarterly_rebalance_on_breach()` (Cell 20)
   - All helper functions
4. Create `utils.py` in the same directory
5. Paste all functions there

---

## Solution 2: Keep Notebook Kernel Running

### Windows PowerShell
```bash
# Terminal 1: Start Jupyter Kernel
jupyter kernel --kernel=python3

# Terminal 2: Run Streamlit (in same Python environment)
streamlit run app.py
```

### Linux/Mac
```bash
# Terminal 1
jupyter kernel

# Terminal 2
streamlit run app.py
```

---

## Solution 3: Convert Notebook to Script

### Using nbconvert (Recommended)
```bash
pip install nbconvert
jupyter nbconvert --to script model.ipynb --stdout > temp_full.py
```

Then manually clean and keep only the function definitions in `utils.py`.

---

## Quickest Setup

1. **Install nbconvert**:
   ```bash
   pip install nbconvert
   ```

2. **Convert notebook to Python**:
   ```bash
   jupyter nbconvert --to script model.ipynb --output utils.py
   ```

3. **Edit utils.py**:
   - Remove all `%` magic commands and plotting code
   - Keep only function definitions
   - Remove data loading code (app.py handles this)

4. **Run app**:
   ```bash
   streamlit run app.py
   ```

---

## What Functions to Keep in utils.py

### Essential Functions (Must Have)
```python
✅ estimate_betas_asof_nifty()
✅ compute_sigma_f_and_lambda()
✅ build_mu_and_sigma_from_betas()
✅ optimize_pulp_mad_targetbetas_cardinality()
✅ compute_achievable_beta_bounds()
✅ compute_conditional_beta_bounds()
✅ backtest_fixed_window_quarterly_rebalance_on_breach()
✅ debug_optimization_failure()  # (if you have it)
```

### Helper Functions (Keep)
```python
✅ get_factor_window_asof()
✅ _to_month_end()
✅ _maybe_percent_to_decimal()
✅ build_portfolio_asof_pulp()
```

### Remove These
```python
❌ %matplotlib inline
❌ !pip install
❌ display() calls
❌ pd.set_option()
❌ Data loading code (pd.read_csv, etc.)
❌ Plotting and visualization code
```

---

## Minimal utils.py Structure

```python
# utils.py - Core Functions for Streamlit App

import pandas as pd
import numpy as np
import pulp
from scipy import stats

# Paste all function definitions here
# Remove notebook-specific code like:
# - %matplotlib commands
# - display() calls  
# - Data loading (app.py handles this)

def estimate_betas_asof_nifty(...):
    # Function implementation
    pass

def compute_sigma_f_and_lambda(...):
    # Function implementation
    pass

# ... other functions ...
```

---

## Testing Your Setup

After creating `utils.py`, test it:

```python
# test_utils.py
from utils import (
    estimate_betas_asof_nifty,
    backtest_fixed_window_quarterly_rebalance_on_breach
)

print("✅ All imports successful!")
```

Run:
```bash
python test_utils.py
```

If successful, Streamlit app will work!

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'utils'"
- **Solution**: Make sure `utils.py` is in the same directory as `app.py`
- **Check**: Run `ls *.py` to verify

### Issue: "ImportError: cannot import name 'backtest_fixed...'"
- **Solution**: Make sure all function definitions are in `utils.py`
- **Check**: `grep -n "def backtest" utils.py` should show the function

### Issue: "NameError: name 'pulp' is not defined"
- **Solution**: Add these imports at top of `utils.py`:
  ```python
  import pandas as pd
  import numpy as np
  import pulp
  from scipy import stats
  ```

---

## Next Steps

1. Create `utils.py` using one of the methods above
2. Test with `python test_utils.py`
3. Run Streamlit: `streamlit run app.py`
4. Open http://localhost:8501 in browser

---

**Need Help?** See DEPLOYMENT_GUIDE.md for more details.
