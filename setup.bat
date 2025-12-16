@echo off
echo ============================================================
echo Viral Post Predictor - Setup
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found
    echo Install from: https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo [4/4] Creating directories...
mkdir data\synthetic 2>nul
mkdir data\processed 2>nul
mkdir models 2>nul

echo.
echo ============================================================
echo ✅ Setup Complete!
echo ============================================================
echo.
echo Next: Generate data
echo   venv\Scripts\activate
echo   python src\generate_data.py
echo.
pause
