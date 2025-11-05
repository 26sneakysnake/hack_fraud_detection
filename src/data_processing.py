"""
Data loading and preprocessing functions
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Tuple, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def load_transactions(filepath: str) -> pd.DataFrame:
    """
    Load transactions data

    Args:
        filepath: Path to transactions CSV file

    Returns:
        DataFrame with transactions
    """
    logger.info(f"Loading transactions from {filepath}")
    df = pd.read_csv(filepath)

    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    logger.info(f"Loaded {len(df)} transactions with {df.shape[1]} features")
    return df


def load_fraud_labels(filepath: str) -> pd.DataFrame:
    """
    Load fraud labels from JSON file

    Args:
        filepath: Path to fraud labels JSON file

    Returns:
        DataFrame with transaction_id and is_fraud columns
    """
    logger.info(f"Loading fraud labels from {filepath}")
    with open(filepath, 'r') as f:
        labels_dict = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame([
        {'transaction_id': tid, 'is_fraud': label}
        for tid, label in labels_dict.items()
    ])

    # Convert transaction_id to appropriate type
    df['transaction_id'] = df['transaction_id'].astype(int)
    df['is_fraud'] = df['is_fraud'].astype(int)

    fraud_rate = df['is_fraud'].mean()
    logger.info(f"Loaded {len(df)} labels. Fraud rate: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")

    return df


def load_cards_data(filepath: str) -> pd.DataFrame:
    """
    Load cards data

    Args:
        filepath: Path to cards CSV file

    Returns:
        DataFrame with card information
    """
    logger.info(f"Loading cards data from {filepath}")
    df = pd.read_csv(filepath)

    # Convert date columns if present
    date_columns = ['card_issue_date', 'card_expiry_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    logger.info(f"Loaded {len(df)} cards with {df.shape[1]} features")
    return df


def load_users_data(filepath: str) -> pd.DataFrame:
    """
    Load users data

    Args:
        filepath: Path to users CSV file

    Returns:
        DataFrame with user information
    """
    logger.info(f"Loading users data from {filepath}")
    df = pd.read_csv(filepath)

    # Convert date columns if present
    if 'account_creation_date' in df.columns:
        df['account_creation_date'] = pd.to_datetime(df['account_creation_date'], errors='coerce')

    logger.info(f"Loaded {len(df)} users with {df.shape[1]} features")
    return df


def load_mcc_codes(filepath: str) -> pd.DataFrame:
    """
    Load MCC (Merchant Category Codes) data

    Args:
        filepath: Path to MCC codes JSON file

    Returns:
        DataFrame with MCC codes and descriptions
    """
    logger.info(f"Loading MCC codes from {filepath}")
    with open(filepath, 'r') as f:
        mcc_dict = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame([
        {'mcc_code': code, 'mcc_description': desc}
        for code, desc in mcc_dict.items()
    ])

    logger.info(f"Loaded {len(df)} MCC codes")
    return df


def merge_all_data(transactions: pd.DataFrame,
                   labels: Optional[pd.DataFrame] = None,
                   cards: Optional[pd.DataFrame] = None,
                   users: Optional[pd.DataFrame] = None,
                   mcc_codes: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Merge all data sources

    Args:
        transactions: Transactions DataFrame
        labels: Fraud labels DataFrame (optional, for training)
        cards: Cards DataFrame
        users: Users DataFrame
        mcc_codes: MCC codes DataFrame

    Returns:
        Merged DataFrame
    """
    logger.info("Merging all data sources...")
    df = transactions.copy()

    # Merge with fraud labels (for training data)
    if labels is not None:
        df = df.merge(labels, on='transaction_id', how='left')
        logger.info(f"Merged with fraud labels. Fraud cases: {df['is_fraud'].sum()}")

    # Merge with cards data
    if cards is not None:
        merge_col = 'card_id' if 'card_id' in df.columns else 'card_number'
        if merge_col in cards.columns:
            df = df.merge(cards, on=merge_col, how='left', suffixes=('', '_card'))
            logger.info(f"Merged with cards data")

    # Merge with users data
    if users is not None:
        user_col = 'user_id' if 'user_id' in df.columns else 'account_id'
        if user_col in users.columns:
            df = df.merge(users, on=user_col, how='left', suffixes=('', '_user'))
            logger.info(f"Merged with users data")

    # Merge with MCC codes
    if mcc_codes is not None:
        if 'mcc' in df.columns or 'mcc_code' in df.columns:
            mcc_col = 'mcc' if 'mcc' in df.columns else 'mcc_code'
            mcc_codes_copy = mcc_codes.copy()
            mcc_codes_copy.columns = [mcc_col if col == 'mcc_code' else col for col in mcc_codes_copy.columns]
            df = df.merge(mcc_codes_copy, on=mcc_col, how='left')
            logger.info(f"Merged with MCC codes")

    logger.info(f"Final merged data shape: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame,
                         strategy: str = 'median',
                         fill_categorical: str = 'unknown') -> pd.DataFrame:
    """
    Handle missing values in the dataset

    Args:
        df: Input DataFrame
        strategy: Strategy for numerical columns ('mean', 'median', 'mode', 'drop')
        fill_categorical: Value to fill categorical columns

    Returns:
        DataFrame with handled missing values
    """
    logger.info("Handling missing values...")
    df = df.copy()

    # Report missing values
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = pd.DataFrame({
        'missing_count': missing,
        'missing_percentage': missing_pct
    })
    missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_percentage', ascending=False)

    if len(missing_df) > 0:
        logger.info(f"Columns with missing values:\n{missing_df}")

    # Handle numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            if strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)

    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(fill_categorical, inplace=True)

    logger.info("Missing values handled successfully")
    return df


def remove_outliers(df: pd.DataFrame,
                   columns: list,
                   method: str = 'iqr',
                   threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers from specified columns

    Args:
        df: Input DataFrame
        columns: List of columns to check for outliers
        method: Method to use ('iqr' or 'zscore')
        threshold: Threshold for outlier detection (IQR multiplier or z-score)

    Returns:
        DataFrame with outliers removed
    """
    logger.info(f"Removing outliers using {method} method...")
    df = df.copy()
    initial_len = len(df)

    for col in columns:
        if col not in df.columns:
            continue

        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores < threshold]

    removed = initial_len - len(df)
    logger.info(f"Removed {removed} outliers ({100*removed/initial_len:.2f}%)")

    return df


def split_temporal(df: pd.DataFrame,
                  date_column: str,
                  split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally (important for time-series data like transactions)

    Args:
        df: Input DataFrame
        date_column: Name of the date column
        split_date: Date to split on (YYYY-MM-DD)

    Returns:
        Tuple of (train_df, validation_df)
    """
    logger.info(f"Splitting data temporally at {split_date}")

    split_timestamp = pd.to_datetime(split_date)
    train_df = df[df[date_column] <= split_timestamp].copy()
    val_df = df[df[date_column] > split_timestamp].copy()

    logger.info(f"Train set: {len(train_df)} samples ({train_df[date_column].min()} to {train_df[date_column].max()})")
    logger.info(f"Validation set: {len(val_df)} samples ({val_df[date_column].min()} to {val_df[date_column].max()})")

    return train_df, val_df


def check_data_leakage(train_df: pd.DataFrame,
                      eval_df: pd.DataFrame,
                      id_column: str = 'user_id') -> Dict[str, any]:
    """
    Check for data leakage between train and evaluation sets

    Args:
        train_df: Training DataFrame
        eval_df: Evaluation DataFrame
        id_column: Column to check for overlap (user_id, card_id, etc.)

    Returns:
        Dictionary with leakage statistics
    """
    if id_column not in train_df.columns or id_column not in eval_df.columns:
        logger.warning(f"Column {id_column} not found in both datasets")
        return {}

    train_ids = set(train_df[id_column].unique())
    eval_ids = set(eval_df[id_column].unique())
    overlap = train_ids.intersection(eval_ids)

    overlap_pct = 100 * len(overlap) / len(eval_ids) if len(eval_ids) > 0 else 0

    stats = {
        'train_unique': len(train_ids),
        'eval_unique': len(eval_ids),
        'overlap_count': len(overlap),
        'overlap_percentage': overlap_pct
    }

    logger.info(f"Data leakage check for {id_column}:")
    logger.info(f"  Train unique: {stats['train_unique']}")
    logger.info(f"  Eval unique: {stats['eval_unique']}")
    logger.info(f"  Overlap: {stats['overlap_count']} ({overlap_pct:.2f}%)")

    return stats
