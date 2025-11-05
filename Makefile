.PHONY: help install train predict dashboard clean test lint format

help:
	@echo "Fraud Detection System - Available Commands"
	@echo ""
	@echo "  make install     - Install all dependencies"
	@echo "  make train       - Train the fraud detection model"
	@echo "  make predict     - Generate predictions on evaluation set"
	@echo "  make dashboard   - Launch the interactive dashboard"
	@echo "  make all         - Run train, predict, and launch dashboard"
	@echo "  make clean       - Clean generated files"
	@echo "  make test        - Run tests (if available)"
	@echo "  make lint        - Run code linting"
	@echo "  make format      - Format code with black"
	@echo ""

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Installation complete!"

train:
	@echo "Training fraud detection model..."
	python scripts/train.py
	@echo "✓ Training complete!"

predict:
	@echo "Generating predictions..."
	python scripts/predict.py
	@echo "✓ Predictions complete!"

dashboard:
	@echo "Launching dashboard..."
	streamlit run dashboard/app.py

all: train predict dashboard

clean:
	@echo "Cleaning generated files..."
	rm -rf models/*.pkl
	rm -rf results/*.csv
	rm -rf results/*.json
	rm -rf data/processed/*.csv
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf .pytest_cache
	rm -f *.log
	@echo "✓ Clean complete!"

test:
	@echo "Running tests..."
	pytest tests/ -v
	@echo "✓ Tests complete!"

lint:
	@echo "Running linter..."
	flake8 src/ scripts/ --max-line-length=120
	@echo "✓ Linting complete!"

format:
	@echo "Formatting code..."
	black src/ scripts/ --line-length=120
	@echo "✓ Formatting complete!"

notebook:
	@echo "Starting Jupyter..."
	jupyter notebook notebooks/

setup:
	@echo "Setting up project structure..."
	mkdir -p data/raw data/processed
	mkdir -p models
	mkdir -p results/figures
	mkdir -p docs/figures docs/certifications
	@echo "✓ Setup complete!"
