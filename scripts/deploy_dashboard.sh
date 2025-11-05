#!/bin/bash

# Fraud Detection Dashboard Deployment Script

echo "=========================================="
echo "Fraud Detection Dashboard Deployment"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

echo "✓ Python 3 found"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt
pip install -q streamlit

# Check if required files exist
echo ""
echo "Checking required files..."

required_files=(
    "results/training_results.json"
    "results/predictions_with_probabilities.csv"
    "data/processed/train_processed.csv"
    "results/feature_importance.csv"
)

missing_files=0

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ Found: $file"
    else
        echo "❌ Missing: $file"
        missing_files=$((missing_files + 1))
    fi
done

if [ $missing_files -gt 0 ]; then
    echo ""
    echo "⚠️  Warning: $missing_files required file(s) missing"
    echo ""
    echo "To generate missing files, run:"
    echo "  python scripts/train.py"
    echo "  python scripts/predict.py"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Launching Dashboard..."
echo "=========================================="
echo ""
echo "Dashboard will be available at:"
echo "  Local:   http://localhost:8501"
echo "  Network: http://$(hostname -I | awk '{print $1}'):8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

# Launch dashboard
streamlit run dashboard/app.py
