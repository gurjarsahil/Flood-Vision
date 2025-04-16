#!/bin/bash

echo "Running Flood Prediction and Mapping Demo"
echo "This will create synthetic data, train a model, and generate flood prediction maps"
echo "------------------------------------------------------------------------------"

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    echo "Please install Python from https://www.python.org/downloads/"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
pip install -r requirements.txt

# Run demo
echo "------------------------------------------------------------------------------"
echo "Starting flood prediction demo..."
python demo.py

if [ $? -ne 0 ]; then
    echo "Error: Demo script failed."
    exit 1
else
    echo "------------------------------------------------------------------------------"
    echo "Demo completed successfully!"
    echo "Check the outputs/demo directory for results."
    echo "------------------------------------------------------------------------------"
fi 