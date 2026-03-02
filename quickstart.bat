@echo off
REM Quick Start Script for Streamlit Deployment
REM Run this from the project directory

echo.
echo ╔═══════════════════════════════════════════════════════════════┐
echo ║  Factor Model Portfolio Optimizer - Quick Start              ║
echo ║  Streamlit Deployment for Windows                            ║
echo ╚═══════════════════════════════════════════════════════════════╝
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+
    echo    https://www.python.org/downloads/
    pause
    exit /b 1
)
echo ✅ Python found

REM Check if virtual environment exists
if not exist "venv" (
    echo.
    echo 📦 Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
)

REM Activate virtual environment
echo.
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✅ Virtual environment activated

REM Upgrade pip first
echo.
echo ⬆️  Upgrading pip...
python -m pip install --quiet --upgrade pip
if errorlevel 1 (
    echo ⚠️  Warning: Could not upgrade pip, continuing anyway...
)

REM Install dependencies with --prefer-binary flag
echo.
echo 📥 Installing dependencies...
pip install --prefer-binary --quiet -r requirements.txt
if errorlevel 1 (
    echo.
    echo ❌ Failed to install dependencies
    echo.
    echo 🔧 Trying alternative installation method...
    pip install --only-binary :all: --quiet -r requirements.txt
    if errorlevel 1 (
        echo ❌ Installation failed
        echo    Please run manually:
        echo    pip install -r requirements.txt --prefer-binary
        pause
        exit /b 1
    )
)
echo ✅ Dependencies installed

REM Setup Streamlit functions
echo.
echo 🔨 Setting up Streamlit app...
python setup_streamlit.py
if errorlevel 1 (
    echo ⚠️  Setup script had issues. Trying to continue...
)

REM Check if utils.py exists
if not exist "utils.py" (
    echo.
    echo ⚠️  WARNING: utils.py not found
    echo    The app may not work until functions are extracted
    echo    See EXTRACT_FUNCTIONS.md for manual setup instructions
    echo.
)

REM Run Streamlit
echo.
echo ╔═══════════════════════════════════════════════════════════════┐
echo ║  ✅ Setup Complete!                                          ║
echo ╠═══════════════════════════════════════════════════════════════╣
echo ║                                                               ║
echo ║  Starting Streamlit app...                                   ║
echo ║  https://localhost:8501                                      ║
echo ║                                                               ║
echo ║  Press Ctrl+C to stop the server                             ║
echo ║                                                               ║
echo ╚═══════════════════════════════════════════════════════════════╝
echo.

streamlit run app.py

pause
