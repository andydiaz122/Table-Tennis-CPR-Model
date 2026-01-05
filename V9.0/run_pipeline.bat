@echo off
:: This batch script runs the complete CPR data processing pipeline.
:: It should be scheduled to run every 30 minutes by the Windows Task Scheduler.

:: Change directory to the script's location to ensure file paths are correct.
cd /d "%~dp0"

:: Clear the screen for a clean run.
cls

echo ==========================================================
echo  CPR Model - Data Pipeline Started at %TIME% on %DATE%
echo ==========================================================
echo.

:: Step 1: Run the data collector
echo [INFO] Step 1 of 3: Running data collector (betsapi_ligapro_RECENT_1HR_collector.py)...
python betsapi_ligapro_RECENT_1HR_collector.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] The data collector script failed. Aborting pipeline.
    goto End
)
echo [SUCCESS] Data collector finished.
echo.

:: Step 2: Run the feature engineering script
echo [INFO] Step 2 of 3: Running advanced feature engineering (advanced_feature_engineering_v7.4.py)...
python advanced_feature_engineering_v7.4.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] The feature engineering script failed. Aborting pipeline.
    goto End
)
echo [SUCCESS] Feature engineering finished.
echo.

:: Step 3: Run the data merging script
echo [INFO] Step 3 of 3: Running data merger (merge_data_v7.4.py)...
python merge_data_v7.4.py
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] The data merge script failed. Aborting pipeline.
    goto End
)
echo [SUCCESS] Data merging finished.
echo.

echo ==========================================================
echo  Pipeline Completed Successfully at %TIME%
echo ==========================================================
echo.

:End
:: The 'pause' command is useful for manual testing.
:: Task Scheduler will ignore it and close the window automatically.
pause
