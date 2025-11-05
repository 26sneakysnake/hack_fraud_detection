# 🚀 Quick Start Guide

Get your fraud detection system up and running in 5 minutes!

## Prerequisites

- Python 3.8+
- pip
- 2GB free disk space
- Your data files

## Installation Steps

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/26sneakysnake/hack_fraud_detection.git
cd hack_fraud_detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Your Data

Place these files in `data/raw/`:

```
data/raw/
├── transactions_train.csv
├── train_fraud_labels.json
├── cards_data.csv
├── users_data.csv
├── mcc_codes.json
└── evaluation_features.csv
```

### 3. Train Model

```bash
python scripts/train.py
```

Expected output:
```
Training fraud detection model...
✓ Data loaded successfully
✓ Features engineered
✓ Model trained
✓ Model saved to models/fraud_detection_model.pkl
```

Training time: ~5-15 minutes depending on your hardware

### 4. Generate Predictions

```bash
python scripts/predict.py
```

Expected output:
```
Making predictions...
✓ Model loaded
✓ Predictions saved to results/submission.csv
```

### 5. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Your dashboard will open at: `http://localhost:8501`

## Quick Commands

Using Makefile (Linux/Mac):

```bash
make install    # Install dependencies
make train      # Train model
make predict    # Generate predictions
make dashboard  # Launch dashboard
make all        # Do everything
```

## Verify Installation

Check if everything is working:

```bash
# Check Python version
python --version  # Should be 3.8+

# Check key packages
python -c "import pandas; import sklearn; import xgboost; print('All packages OK')"

# List files
ls data/raw/  # Should show your data files
```

## Troubleshooting

### Problem: Import errors
**Solution**: Reinstall dependencies
```bash
pip install -r requirements.txt --force-reinstall
```

### Problem: Data not found
**Solution**: Check file paths in `config/config.yaml`

### Problem: Out of memory
**Solution**: Reduce data size or increase system RAM

### Problem: Training is slow
**Solution**: Use fewer models or reduce hyperparameter search space in config

## What's Next?

1. **Explore the Dashboard**
   - View model performance
   - Analyze predictions
   - Search transactions

2. **Review Notebooks**
   - Open `notebooks/01_EDA.ipynb`
   - Understand the data analysis

3. **Customize the Model**
   - Edit `config/config.yaml`
   - Try different models
   - Tune hyperparameters

4. **Deploy to Production**
   - Set up IBM watsonx credentials
   - Enable watsonx in config
   - Deploy model

## Resources

- **Full Documentation**: See [README.md](README.md)
- **Dashboard Guide**: See [dashboard/README.md](dashboard/README.md)
- **Data Guide**: See [data/README.md](data/README.md)
- **Report Template**: See [docs/REPORT_TEMPLATE.md](docs/REPORT_TEMPLATE.md)

## Support

Need help?
- Check [CONTRIBUTING.md](CONTRIBUTING.md)
- Open an issue on GitHub
- Review error logs: `training.log` and `prediction.log`

## Success Checklist

- [ ] Repository cloned
- [ ] Dependencies installed
- [ ] Data files added
- [ ] Model trained successfully
- [ ] Predictions generated
- [ ] Dashboard running
- [ ] Results look good

**Congratulations! Your fraud detection system is ready! 🎉**

---

Time to completion: ~10 minutes
Difficulty: Easy ⭐
