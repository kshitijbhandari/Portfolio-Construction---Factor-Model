# ✅ Windows Installation Fix

If you get numpy/pandas build errors, use this manual installation instead:

## Option 1: Direct Command (Easiest)

```powershell
# Open PowerShell in your project directory and run:

python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install scipy==1.11.1
pip install matplotlib==3.7.2
pip install streamlit==1.28.1
pip install plotly==5.17.0
pip install pulp==2.7.0
pip install openpyxl==3.1.2
```

Then run:
```powershell
python setup_streamlit.py
streamlit run app.py
```

---

## Option 2: Simpler Requirements File

Replace `requirements.txt` with `requirements_simple.txt`:
```bash
pip install -r requirements_simple.txt
```

---

## Option 3: Install One at a Time (Most Reliable)

```powershell
pip install --no-cache-dir --prefer-binary numpy==1.24.3
pip install --no-cache-dir --prefer-binary pandas==2.0.3
pip install --no-cache-dir --prefer-binary scipy==1.11.1
pip install --no-cache-dir --prefer-binary matplotlib==3.7.2
pip install --no-cache-dir --prefer-binary streamlit==1.28.1
```

---

## Why This Happens

The error occurs because:
1. NumPy 1.26+ requires a C compiler to build
2. Visual Studio Build Tools aren't installed
3. The --prefer-binary flag finds no pre-built wheels

**Solution**: Use older versions (numpy 1.24.3, pandas 2.0.3) that have pre-built wheels.

---

## Quick Commands to Copy & Paste

### For PowerShell:
```powershell
cd c:\Users\kshit\Personal_Factor_model
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_simple.txt
python setup_streamlit.py
streamlit run app.py
```

### For Command Prompt (cmd.exe):
```batch
cd c:\Users\kshit\Personal_Factor_model
python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements_simple.txt
python setup_streamlit.py
streamlit run app.py
```

---

## Troubleshooting

**If you still get build errors:**
1. Remove the venv folder: `Remove-Item -Path venv -Recurse -Force`
2. Create fresh venv: `python -m venv venv`
3. Activate: `.\venv\Scripts\Activate.ps1`
4. Install pip upgrade: `python -m pip install --upgrade pip`
5. Try single package: `pip install --no-cache-dir numpy==1.24.3`

**If specific package fails:**
- Try an older version (e.g., pandas==2.0.2 instead of 2.0.3)
- Or skip to next package and come back

---

## You're Almost There!

Once installation completes, you should see:
```
✅ Successfully installed numpy-1.24.3
✅ Successfully installed pandas-2.0.3
... etc
```

Then just run:
```powershell
streamlit run app.py
```

And open: **http://localhost:8501**

---

Need help? Contact Python support or check: https://stackoverflow.com/questions/tagged/numpy+build-error+windows
