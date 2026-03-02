#!/usr/bin/env python
"""
Extract functions from model.ipynb and create utils.py for Streamlit app
Usage: python setup_streamlit.py
"""

import json
import sys
from pathlib import Path

def extract_functions_from_notebook():
    """Extract function definitions from model.ipynb"""
    
    notebook_path = Path("model.ipynb")
    
    if not notebook_path.exists():
        print("❌ Error: model.ipynb not found in current directory")
        print("   Run this script from the same directory as model.ipynb")
        return False
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"❌ Error reading notebook: {e}")
        return False
    
    # Extract code cells that contain function definitions
    function_code = []
    
    # Add imports
    function_code.append("""# Auto-generated from model.ipynb
# Core functions for factor model portfolio optimization

import pandas as pd
import numpy as np
import pulp
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

""")
    
    # Track what we've extracted
    extracted_functions = []
    
    for cell in notebook.get('cells', []):
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell['source'])
        
        # Skip cells with magic commands or data loading
        if any(x in source for x in ['%matplotlib', '!pip', 'pd.read_csv', 
                                       'pd.set_option', 'display(', 'plt.show']):
            continue
        
        # Keep cells that define functions
        if 'def ' in source:
            # Clean up notebook-specific syntax
            source = source.replace('%', '#%')  # comment out magic commands
            function_code.append(source)
            function_code.append("\n\n")
            
            # Extract function names
            for line in source.split('\n'):
                if line.strip().startswith('def '):
                    func_name = line.split('(')[0].replace('def ', '').strip()
                    extracted_functions.append(func_name)
    
    # Check if we got key functions
    required_functions = [
        'estimate_betas_asof_nifty',
        'backtest_fixed_window_quarterly_rebalance_on_breach',
        'optimize_pulp_mad_targetbetas_cardinality'
    ]
    
    found = [f for f in extracted_functions if any(req in f for req in required_functions)]
    
    if len(found) < 2:
        print(f"⚠️  Warning: Only found {len(found)} of {len(required_functions)} required functions")
        print(f"   Found: {found}")
    
    # Write utils.py
    utils_path = Path("utils.py")
    try:
        with open(utils_path, 'w', encoding='utf-8') as f:
            f.writelines(function_code)
        print(f"✅ Successfully created utils.py")
        print(f"   Functions extracted: {', '.join(extracted_functions[:5])}...")
        return True
    except Exception as e:
        print(f"❌ Error writing utils.py: {e}")
        return False

def verify_imports():
    """Verify that utils.py can be imported"""
    try:
        import utils
        print("✅ utils.py import successful")
        
        # Check for key functions
        key_funcs = [
            'estimate_betas_asof_nifty',
            'backtest_fixed_window_quarterly_rebalance_on_breach'
        ]
        
        for func in key_funcs:
            if hasattr(utils, func):
                print(f"   ✅ Found {func}")
            else:
                print(f"   ❌ Missing {func}")
                return False
        
        return True
    except ImportError as e:
        print(f"❌ Cannot import utils: {e}")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    required = ['pandas', 'numpy', 'pulp', 'streamlit', 'matplotlib']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        print(f"   Run: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages installed")
    return True

def main():
    print("=" * 60)
    print("🚀 Streamlit Setup for Factor Model Portfolio Optimizer")
    print("=" * 60)
    print()
    
    # Step 1: Check dependencies
    print("Step 1: Checking dependencies...")
    if not check_dependencies():
        print("   Please install missing packages first")
        return False
    print()
    
    # Step 2: Extract functions
    print("Step 2: Extracting functions from notebook...")
    if not extract_functions_from_notebook():
        print("   Failed to extract functions")
        return False
    print()
    
    # Step 3: Verify imports
    print("Step 3: Verifying imports...")
    if not verify_imports():
        print("   Import verification failed")
        print("   You may need to manually copy functions to utils.py")
        print("   See EXTRACT_FUNCTIONS.md for manual instructions")
        return False
    print()
    
    # Success!
    print("=" * 60)
    print("✅ Setup complete! You can now run:")
    print()
    print("   streamlit run app.py")
    print()
    print("This will open the app at http://localhost:8501")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
