# Data Directory

This directory contains all datasets for the fraud detection project.

## Directory Structure

```
data/
├── raw/                    # Raw, unprocessed data files
│   ├── transactions_train.csv
│   ├── train_fraud_labels.json
│   ├── cards_data.csv
│   ├── users_data.csv
│   ├── mcc_codes.json
│   └── evaluation_features.csv
│
└── processed/              # Processed and feature-engineered data
    ├── train_processed.csv
    ├── eval_processed.csv
    └── eda_data.csv
```

## Raw Data Files

### transactions_train.csv
Training transaction data with the following expected columns:
- `transaction_id`: Unique transaction identifier
- `timestamp`: Transaction timestamp
- `amount`: Transaction amount
- `user_id`: User/account identifier
- `card_id` or `card_number`: Card identifier
- `mcc`: Merchant Category Code
- Additional transaction features...

**Size**: ~210,000 transactions (2016-2018)

### train_fraud_labels.json
Fraud labels for training transactions in JSON format:
```json
{
    "transaction_id_1": 0,
    "transaction_id_2": 1,
    ...
}
```

Where:
- `0` = Not Fraud
- `1` = Fraud

### cards_data.csv
Card information including:
- `card_id`: Card identifier
- `card_type`: Type of card
- `card_issue_date`: When the card was issued
- `card_expiry_date`: Card expiration date
- Additional card metadata...

### users_data.csv
User/account information including:
- `user_id`: User identifier
- `account_creation_date`: Account creation timestamp
- `user_type`: Type of user/account
- Additional user demographics...

### mcc_codes.json
Merchant Category Codes dictionary in JSON format:
```json
{
    "5411": "Grocery Stores, Supermarkets",
    "5812": "Eating Places, Restaurants",
    ...
}
```

### evaluation_features.csv
Evaluation dataset for predictions (IMPORTANT: No labels provided)
- Same structure as `transactions_train.csv`
- Used only for final predictions
- **DO NOT use for training or validation**

## Processed Data Files

### train_processed.csv
Fully processed training data with:
- All engineered features
- Merged information from all sources
- Handled missing values
- Ready for model training

### eval_processed.csv
Processed evaluation data with same features as training data

### eda_data.csv
Data output from exploratory analysis with initial processing

## Data Loading

Use the provided functions in `src/data_processing.py`:

```python
from src.data_processing import (
    load_transactions,
    load_fraud_labels,
    load_cards_data,
    load_users_data,
    load_mcc_codes,
    merge_all_data
)

# Load all data
transactions = load_transactions('data/raw/transactions_train.csv')
labels = load_fraud_labels('data/raw/train_fraud_labels.json')
cards = load_cards_data('data/raw/cards_data.csv')
users = load_users_data('data/raw/users_data.csv')
mcc_codes = load_mcc_codes('data/raw/mcc_codes.json')

# Merge everything
df = merge_all_data(transactions, labels, cards, users, mcc_codes)
```

## Important Notes

⚠️ **Data Privacy**: Raw data files are not included in version control (see `.gitignore`)

⚠️ **Data Leakage**:
- Never use `evaluation_features.csv` for training
- Use temporal split for validation (2016-2017 train, 2018 validation)
- Check for user/card overlap between train and evaluation sets

⚠️ **Missing Values**: Some datasets may have missing values. Use the provided preprocessing functions to handle them appropriately.

## Data Size

Expected sizes (approximate):
- `transactions_train.csv`: ~50-100 MB
- `evaluation_features.csv`: ~25-50 MB
- `cards_data.csv`: ~1-5 MB
- `users_data.csv`: ~1-5 MB
- `mcc_codes.json`: <1 MB
- `train_fraud_labels.json`: <1 MB

## Adding New Data

To add your data files:

1. Place all raw data files in `data/raw/`
2. Ensure file names match the configuration in `config/config.yaml`
3. Run the training pipeline: `python scripts/train.py`

## Data Quality Checks

Before training, verify:
- [ ] All files are present in `data/raw/`
- [ ] File formats match expectations (CSV/JSON)
- [ ] No empty files
- [ ] Transaction IDs match between transactions and labels
- [ ] Timestamps are parseable
- [ ] No obvious data corruption

## Support

For data-related issues, check:
1. File paths in `config/config.yaml`
2. Data loading functions in `src/data_processing.py`
3. Error logs in `training.log` or `prediction.log`
