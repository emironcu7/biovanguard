@echo off
REM BioVanguard Startup Script for Windows
REM This script activates the virtual environment and starts the server

echo.
echo ============================================================
echo   BioVanguard - AI-Powered Drug Discovery Platform
echo   Target: c-Myc Protein
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist ".venv310\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run the following commands first:
    echo.
    echo   py -3.10 -m venv .venv310
    echo   .\.venv310\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/3] Activating virtual environment...
call .venv310\Scripts\activate.bat

REM Check if models exist
if not exist "PredictionPipeline\gnn_model_final.pt" (
    echo.
    echo [WARNING] Model files not found in PredictionPipeline/
    echo Please ensure all required files are present:
    echo   - gnn_model_final.pt
    echo   - xgb_model_final.joblib
    echo   - xgb_scaler.joblib
    echo   - receptor.pdb
    echo   - receptor.pdbqt
    echo.
    pause
)

REM Create necessary directories
echo [2/3] Creating directories...
if not exist "uploads" mkdir uploads
if not exist "web_runs" mkdir web_runs

REM Start server
echo [3/3] Starting BioVanguard server...
echo.
echo Server will be available at: http://127.0.0.1:8000
echo Press Ctrl+C to stop the server
echo.

python server.py --host 127.0.0.1 --port 8000

REM If server stops
echo.
echo Server stopped.
pause
