"""
Prediction script for fraud detection model
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logging, create_submission_file, load_json
from src.data_processing import (
    load_transactions,
    load_cards_data,
    load_users_data,
    load_mcc_codes,
    merge_all_data,
    handle_missing_values
)
from src.feature_engineering import create_all_features
from src.models import FraudDetectionModel

# Setup logging
logger = setup_logging('prediction.log', level=logging.INFO)


def main(config_path='config/config.yaml', model_path='models/fraud_detection_model.pkl'):
    """
    Main prediction pipeline

    Args:
        config_path: Path to configuration file
        model_path: Path to saved model
    """
    logger.info("=" * 50)
    logger.info("FRAUD DETECTION PREDICTIONS")
    logger.info("=" * 50)

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(config_path)

    # Load training results for feature list and threshold
    training_results = load_json('results/training_results.json')
    feature_cols = training_results['features']
    optimal_threshold = training_results['optimal_threshold']

    logger.info(f"Using threshold: {optimal_threshold:.4f}")
    logger.info(f"Number of features: {len(feature_cols)}")

    # Load model
    logger.info("\n" + "=" * 50)
    logger.info("LOADING MODEL")
    logger.info("=" * 50)

    model = FraudDetectionModel.load(model_path)

    # Load evaluation data
    logger.info("\n" + "=" * 50)
    logger.info("LOADING EVALUATION DATA")
    logger.info("=" * 50)

    eval_transactions = load_transactions(config['data']['evaluation_features'])

    # Load auxiliary data (same as training)
    try:
        cards = load_cards_data(config['data']['cards_data'])
        users = load_users_data(config['data']['users_data'])
        mcc_codes = load_mcc_codes(config['data']['mcc_codes'])

        # Merge data
        logger.info("\nMerging data sources...")
        df_eval = merge_all_data(eval_transactions, None, cards, users, mcc_codes)
    except:
        logger.warning("Could not load auxiliary data, using transactions only")
        df_eval = eval_transactions

    # Save transaction IDs
    transaction_ids = df_eval['transaction_id'].copy()

    # Handle missing values
    logger.info("\nHandling missing values...")
    df_eval = handle_missing_values(df_eval)

    # Feature engineering
    logger.info("\n" + "=" * 50)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 50)

    df_eval = create_all_features(df_eval, is_training=False, config=config)

    # Prepare features (must match training features)
    # Get available features
    available_features = [col for col in feature_cols if col in df_eval.columns]
    missing_features = [col for col in feature_cols if col not in df_eval.columns]

    if missing_features:
        logger.warning(f"Missing {len(missing_features)} features from training:")
        logger.warning(f"First 10: {missing_features[:10]}")

        # Add missing features with zeros
        for col in missing_features:
            df_eval[col] = 0

    # Select features in the same order as training
    X_eval = df_eval[feature_cols].select_dtypes(include=[np.number])

    logger.info(f"Evaluation set shape: {X_eval.shape}")

    # Make predictions
    logger.info("\n" + "=" * 50)
    logger.info("MAKING PREDICTIONS")
    logger.info("=" * 50)

    # Get probabilities
    y_pred_proba = model.predict_proba(X_eval)[:, 1]

    # Apply threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

    logger.info(f"Total predictions: {len(y_pred)}")
    logger.info(f"Predicted fraud: {y_pred.sum()} ({100*y_pred.mean():.2f}%)")

    # Create submission file
    logger.info("\n" + "=" * 50)
    logger.info("CREATING SUBMISSION FILE")
    logger.info("=" * 50)

    submission_path = config['data']['submission']
    create_submission_file(y_pred, transaction_ids, submission_path)

    # Save probabilities for analysis
    results_df = pd.DataFrame({
        'transaction_id': transaction_ids,
        'fraud_probability': y_pred_proba,
        'fraud_prediction': y_pred
    })
    results_df.to_csv('results/predictions_with_probabilities.csv', index=False)
    logger.info("Detailed predictions saved to results/predictions_with_probabilities.csv")

    logger.info("\n" + "=" * 50)
    logger.info("PREDICTION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make fraud predictions')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default='models/fraud_detection_model.pkl',
                       help='Path to saved model')
    args = parser.parse_args()

    try:
        main(args.config, args.model)
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        sys.exit(1)
