"""
Utility functions for the fraud detection system
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration

    Args:
        log_file: Path to log file (optional)
        level: Logging level

    Returns:
        Logger instance
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save dictionary to JSON file

    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def memory_usage_mb(df: pd.DataFrame) -> float:
    """
    Calculate memory usage of DataFrame in MB

    Args:
        df: Pandas DataFrame

    Returns:
        Memory usage in MB
    """
    return df.memory_usage(deep=True).sum() / 1024**2


def optimize_dtypes(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by converting dtypes

    Args:
        df: Pandas DataFrame
        verbose: Print memory savings

    Returns:
        Optimized DataFrame
    """
    start_mem = memory_usage_mb(df)

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = memory_usage_mb(df)

    if verbose:
        print(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')

    return df


def create_submission_file(predictions: np.ndarray,
                          transaction_ids: pd.Series,
                          output_path: str = "results/submission.csv") -> None:
    """
    Create submission file with predictions

    Args:
        predictions: Array of predictions (0 or 1)
        transaction_ids: Series of transaction IDs
        output_path: Path to save submission file
    """
    submission = pd.DataFrame({
        'transaction_id': transaction_ids,
        'fraud_prediction': predictions
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")
    print(f"Total predictions: {len(submission)}")
    print(f"Fraud predictions: {predictions.sum()} ({100*predictions.mean():.2f}%)")


def get_timestamp() -> str:
    """
    Get current timestamp as string

    Returns:
        Timestamp string (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(directory: str) -> None:
    """
    Ensure directory exists, create if it doesn't

    Args:
        directory: Directory path
    """
    os.makedirs(directory, exist_ok=True)
