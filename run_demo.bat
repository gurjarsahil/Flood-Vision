@echo off
echo Running Flood Prediction and Mapping Demo
echo This will create synthetic data, train a model, and generate flood prediction maps
echo ------------------------------------------------------------------------------

REM Check if Python is installed
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    goto :end
)

REM Check if required packages are installed
echo Checking dependencies...
pip install -r requirements.txt

REM Run demo
echo ------------------------------------------------------------------------------
echo Starting flood prediction demo...
python demo.py

if %errorlevel% neq 0 (
    echo Error: Demo script failed.
    goto :end
) else (
    echo ------------------------------------------------------------------------------
    echo Demo completed successfully!
    echo Check the outputs/demo directory for results.
    echo ------------------------------------------------------------------------------
)

:end
pause 