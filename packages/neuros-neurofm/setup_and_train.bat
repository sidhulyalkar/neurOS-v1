@echo off
REM NeuroFM-X Setup and Training Script for Windows
REM This script creates a conda environment, downloads data, and trains the model

echo ================================================================================
echo NeuroFM-X Complete Setup and Training
echo ================================================================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda not found! Please install Anaconda or Miniconda first.
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo [1/4] Creating conda environment...
echo.
conda env create -f environment.yml -y
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create environment. See error above.
    pause
    exit /b 1
)

echo.
echo [2/4] Activating environment...
echo.
call conda activate neurofm
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate environment.
    echo Try running manually: conda activate neurofm
    pause
    exit /b 1
)

echo.
echo [3/4] Downloading Allen Brain Observatory data...
echo This will download ~5-10 GB of data and may take 15-30 minutes.
echo.
python download_allen_data.py --num-sessions 5
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Data download failed. You can try running it manually later:
    echo   conda activate neurofm
    echo   python download_allen_data.py --num-sessions 5
    pause
    exit /b 1
)

echo.
echo [4/4] Starting model training...
echo This will take approximately 2-4 hours on RTX 3070 Ti.
echo.
python train_allen_data.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Training failed. Check error above.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo ALL DONE!
echo ================================================================================
echo.
echo Trained model saved to: checkpoints_allen\best.pt
echo.
pause
