"""
Feature engineering functions for fraud detection
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def extract_temporal_features(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Extract temporal features from timestamp

    Args:
        df: Input DataFrame
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with added temporal features
    """
    logger.info("Extracting temporal features...")
    df = df.copy()

    if timestamp_col not in df.columns:
        logger.warning(f"Column {timestamp_col} not found")
        return df

    # Ensure datetime format
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Extract temporal components
    df['hour'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['day_of_month'] = df[timestamp_col].dt.day
    df['month'] = df[timestamp_col].dt.month
    df['year'] = df[timestamp_col].dt.year
    df['quarter'] = df[timestamp_col].dt.quarter

    # Create derived features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['day_of_week'] < 5)).astype(int)

    # Cyclical encoding for hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Cyclical encoding for day of week
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Cyclical encoding for month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    logger.info(f"Added {15} temporal features")
    return df


def create_user_aggregation_features(df: pd.DataFrame,
                                     user_col: str = 'user_id',
                                     amount_col: str = 'amount',
                                     timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Create user-level aggregation features

    Args:
        df: Input DataFrame
        user_col: User identifier column
        amount_col: Transaction amount column
        timestamp_col: Timestamp column

    Returns:
        DataFrame with added user aggregation features
    """
    logger.info("Creating user aggregation features...")
    df = df.copy()

    if user_col not in df.columns:
        logger.warning(f"Column {user_col} not found")
        return df

    # Sort by user and timestamp
    if timestamp_col in df.columns:
        df = df.sort_values([user_col, timestamp_col])

    # User transaction count
    df['user_transaction_count'] = df.groupby(user_col)[user_col].transform('count')

    if amount_col in df.columns:
        # User amount statistics
        df['user_amount_mean'] = df.groupby(user_col)[amount_col].transform('mean')
        df['user_amount_std'] = df.groupby(user_col)[amount_col].transform('std').fillna(0)
        df['user_amount_median'] = df.groupby(user_col)[amount_col].transform('median')
        df['user_amount_min'] = df.groupby(user_col)[amount_col].transform('min')
        df['user_amount_max'] = df.groupby(user_col)[amount_col].transform('max')

        # Amount deviation from user's average
        df['amount_deviation_from_user_mean'] = df[amount_col] - df['user_amount_mean']
        df['amount_deviation_ratio'] = df['amount_deviation_from_user_mean'] / (df['user_amount_std'] + 1e-6)

        # Z-score of amount for user
        df['user_amount_zscore'] = (df[amount_col] - df['user_amount_mean']) / (df['user_amount_std'] + 1e-6)

    if timestamp_col in df.columns:
        # Time since first transaction
        df['user_first_transaction'] = df.groupby(user_col)[timestamp_col].transform('min')
        df['days_since_first_transaction'] = (df[timestamp_col] - df['user_first_transaction']).dt.total_seconds() / 86400

        # Transaction velocity (transactions per day)
        df['user_transaction_velocity'] = df['user_transaction_count'] / (df['days_since_first_transaction'] + 1)

    logger.info(f"Added user aggregation features")
    return df


def create_card_features(df: pd.DataFrame,
                        card_col: str = 'card_id',
                        timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Create card-level features

    Args:
        df: Input DataFrame
        card_col: Card identifier column
        timestamp_col: Timestamp column

    Returns:
        DataFrame with added card features
    """
    logger.info("Creating card features...")
    df = df.copy()

    if card_col not in df.columns:
        logger.warning(f"Column {card_col} not found")
        return df

    # Card transaction count
    df['card_transaction_count'] = df.groupby(card_col)[card_col].transform('count')

    if timestamp_col in df.columns:
        # Card age
        df['card_first_transaction'] = df.groupby(card_col)[timestamp_col].transform('min')
        df['card_age_days'] = (df[timestamp_col] - df['card_first_transaction']).dt.total_seconds() / 86400

        # Time since last transaction on this card
        df['card_last_transaction'] = df.groupby(card_col)[timestamp_col].transform(
            lambda x: x.shift(1)
        )
        df['hours_since_last_card_transaction'] = (
            df[timestamp_col] - df['card_last_transaction']
        ).dt.total_seconds() / 3600
        df['hours_since_last_card_transaction'].fillna(999, inplace=True)

    logger.info(f"Added card features")
    return df


def create_mcc_features(df: pd.DataFrame,
                       mcc_col: str = 'mcc',
                       amount_col: str = 'amount') -> pd.DataFrame:
    """
    Create merchant category code (MCC) features

    Args:
        df: Input DataFrame
        mcc_col: MCC column name
        amount_col: Amount column name

    Returns:
        DataFrame with added MCC features
    """
    logger.info("Creating MCC features...")
    df = df.copy()

    if mcc_col not in df.columns:
        logger.warning(f"Column {mcc_col} not found")
        return df

    # MCC transaction count
    df['mcc_transaction_count'] = df.groupby(mcc_col)[mcc_col].transform('count')

    # MCC frequency (what percentage of all transactions)
    df['mcc_frequency'] = df['mcc_transaction_count'] / len(df)

    if amount_col in df.columns:
        # MCC amount statistics
        df['mcc_amount_mean'] = df.groupby(mcc_col)[amount_col].transform('mean')
        df['mcc_amount_std'] = df.groupby(mcc_col)[amount_col].transform('std').fillna(0)

        # Amount deviation from MCC average
        df['amount_deviation_from_mcc_mean'] = df[amount_col] - df['mcc_amount_mean']

    logger.info(f"Added MCC features")
    return df


def create_user_mcc_features(df: pd.DataFrame,
                            user_col: str = 'user_id',
                            mcc_col: str = 'mcc') -> pd.DataFrame:
    """
    Create user-MCC interaction features

    Args:
        df: Input DataFrame
        user_col: User identifier column
        mcc_col: MCC column

    Returns:
        DataFrame with user-MCC features
    """
    logger.info("Creating user-MCC interaction features...")
    df = df.copy()

    if user_col not in df.columns or mcc_col not in df.columns:
        logger.warning("Required columns not found")
        return df

    # User's transaction count per MCC
    df['user_mcc_transaction_count'] = df.groupby([user_col, mcc_col])[user_col].transform('count')

    # User's unique MCC count
    df['user_unique_mcc_count'] = df.groupby(user_col)[mcc_col].transform('nunique')

    # Is this a new MCC for this user?
    df['is_new_mcc_for_user'] = (df['user_mcc_transaction_count'] == 1).astype(int)

    logger.info(f"Added user-MCC interaction features")
    return df


def create_behavioral_features(df: pd.DataFrame,
                               user_col: str = 'user_id',
                               amount_col: str = 'amount',
                               timestamp_col: str = 'timestamp',
                               window_days: int = 30) -> pd.DataFrame:
    """
    Create behavioral features based on recent history

    Args:
        df: Input DataFrame
        user_col: User identifier column
        amount_col: Amount column
        timestamp_col: Timestamp column
        window_days: Number of days to look back

    Returns:
        DataFrame with behavioral features
    """
    logger.info(f"Creating behavioral features (window: {window_days} days)...")
    df = df.copy()

    required_cols = [user_col, timestamp_col]
    if not all(col in df.columns for col in required_cols):
        logger.warning("Required columns not found")
        return df

    # Sort by user and timestamp
    df = df.sort_values([user_col, timestamp_col])

    # Transaction in last N days
    window_seconds = window_days * 86400

    # Time since previous transaction (seconds)
    df['time_since_prev_transaction'] = df.groupby(user_col)[timestamp_col].diff().dt.total_seconds()
    df['time_since_prev_transaction'].fillna(999999, inplace=True)

    # Transaction frequency features (would require more complex rolling window implementation)
    # For simplicity, we'll use simpler versions

    if amount_col in df.columns:
        # Rolling statistics (simplified - using expanding window)
        df['rolling_amount_mean'] = df.groupby(user_col)[amount_col].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        df['rolling_amount_std'] = df.groupby(user_col)[amount_col].transform(
            lambda x: x.shift(1).expanding().std()
        )

        # Fill NaN values
        df['rolling_amount_mean'].fillna(df[amount_col].median(), inplace=True)
        df['rolling_amount_std'].fillna(0, inplace=True)

        # Amount deviation from rolling average
        df['amount_vs_rolling_mean'] = df[amount_col] - df['rolling_amount_mean']
        df['amount_vs_rolling_std'] = df['amount_vs_rolling_mean'] / (df['rolling_amount_std'] + 1e-6)

    logger.info(f"Added behavioral features")
    return df


def create_amount_bins(df: pd.DataFrame,
                      amount_col: str = 'amount',
                      n_bins: int = 10) -> pd.DataFrame:
    """
    Create binned features for amount

    Args:
        df: Input DataFrame
        amount_col: Amount column name
        n_bins: Number of bins

    Returns:
        DataFrame with binned amount features
    """
    logger.info(f"Creating amount bins ({n_bins} bins)...")
    df = df.copy()

    if amount_col not in df.columns:
        logger.warning(f"Column {amount_col} not found")
        return df

    # Create bins using quantiles
    df['amount_bin'] = pd.qcut(df[amount_col], q=n_bins, labels=False, duplicates='drop')

    # Create log-scaled amount
    df['amount_log'] = np.log1p(df[amount_col])

    # Create square root scaled amount
    df['amount_sqrt'] = np.sqrt(df[amount_col])

    logger.info(f"Added amount binning features")
    return df


def create_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create risk score features based on domain knowledge

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with risk score features
    """
    logger.info("Creating risk score features...")
    df = df.copy()

    # High-risk MCC categories (examples - adjust based on actual data)
    high_risk_mccs = [5967, 5499, 5999, 7995, 7922, 7273]  # Online/digital goods, travel
    if 'mcc' in df.columns:
        df['is_high_risk_mcc'] = df['mcc'].isin(high_risk_mccs).astype(int)

    # High-risk time periods
    if 'hour' in df.columns:
        df['is_high_risk_hour'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)

    # High amount flag
    if 'amount' in df.columns:
        amount_95th = df['amount'].quantile(0.95)
        df['is_high_amount'] = (df['amount'] > amount_95th).astype(int)

    # Multiple rapid transactions
    if 'time_since_prev_transaction' in df.columns:
        df['is_rapid_transaction'] = (df['time_since_prev_transaction'] < 300).astype(int)  # < 5 minutes

    # New user risk
    if 'days_since_first_transaction' in df.columns:
        df['is_new_user'] = (df['days_since_first_transaction'] < 7).astype(int)

    # Composite risk score
    risk_features = [col for col in df.columns if col.startswith('is_high_') or col.startswith('is_new_')]
    if risk_features:
        df['composite_risk_score'] = df[risk_features].sum(axis=1)

    logger.info(f"Added risk score features")
    return df


def encode_categorical_features(df: pd.DataFrame,
                                categorical_cols: Optional[List[str]] = None,
                                method: str = 'label') -> pd.DataFrame:
    """
    Encode categorical features

    Args:
        df: Input DataFrame
        categorical_cols: List of categorical columns (if None, auto-detect)
        method: Encoding method ('label', 'onehot', 'frequency')

    Returns:
        DataFrame with encoded categorical features
    """
    logger.info(f"Encoding categorical features using {method} encoding...")
    df = df.copy()

    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # Exclude ID columns and description columns
        categorical_cols = [col for col in categorical_cols
                          if not col.endswith('_id')
                          and not col.endswith('_description')
                          and col != 'transaction_id']

    for col in categorical_cols:
        if col not in df.columns:
            continue

        if method == 'label':
            # Label encoding
            df[f'{col}_encoded'] = pd.Categorical(df[col]).codes

        elif method == 'frequency':
            # Frequency encoding
            freq = df[col].value_counts(normalize=True)
            df[f'{col}_frequency'] = df[col].map(freq)

        elif method == 'onehot':
            # One-hot encoding (only for low cardinality)
            if df[col].nunique() <= 20:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)

    logger.info(f"Encoded {len(categorical_cols)} categorical features")
    return df


def create_all_features(df: pd.DataFrame,
                       is_training: bool = True,
                       config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Create all features in the correct order

    Args:
        df: Input DataFrame
        is_training: Whether this is training data
        config: Configuration dictionary

    Returns:
        DataFrame with all features
    """
    logger.info("Creating all features...")
    logger.info(f"Input shape: {df.shape}")

    # Extract temporal features
    df = extract_temporal_features(df)

    # Create user features
    if 'user_id' in df.columns:
        df = create_user_aggregation_features(df)

    # Create card features
    if 'card_id' in df.columns or 'card_number' in df.columns:
        card_col = 'card_id' if 'card_id' in df.columns else 'card_number'
        df = create_card_features(df, card_col=card_col)

    # Create MCC features
    if 'mcc' in df.columns:
        df = create_mcc_features(df)
        if 'user_id' in df.columns:
            df = create_user_mcc_features(df)

    # Create behavioral features
    if 'user_id' in df.columns and 'timestamp' in df.columns:
        window_days = config.get('features', {}).get('window_days', 30) if config else 30
        df = create_behavioral_features(df, window_days=window_days)

    # Create amount bins
    if 'amount' in df.columns:
        df = create_amount_bins(df)

    # Create risk scores
    df = create_risk_scores(df)

    # Encode categorical features
    df = encode_categorical_features(df, method='label')

    logger.info(f"Output shape: {df.shape}")
    logger.info(f"Created {df.shape[1] - df.shape[1]} new features")

    return df
