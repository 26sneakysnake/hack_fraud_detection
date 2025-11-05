"""
Main training script for fraud detection model
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

from src.utils import load_config, setup_logging, save_json
from src.data_processing import (
    load_transactions,
    load_fraud_labels,
    load_cards_data,
    load_users_data,
    load_mcc_codes,
    merge_all_data,
    handle_missing_values,
    split_temporal
)
from src.feature_engineering import create_all_features
from src.models import (
    FraudDetectionModel,
    handle_imbalanced_data,
    tune_hyperparameters,
    train_multiple_models,
    select_best_model
)
from src.evaluation import evaluate_model, find_optimal_threshold

# Setup logging
logger = setup_logging('training.log', level=logging.INFO)


def main(config_path='config/config.yaml'):
    """
    Main training pipeline

    Args:
        config_path: Path to configuration file
    """
    logger.info("=" * 50)
    logger.info("FRAUD DETECTION MODEL TRAINING")
    logger.info("=" * 50)

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(config_path)

    # Load data
    logger.info("\n" + "=" * 50)
    logger.info("LOADING DATA")
    logger.info("=" * 50)

    transactions = load_transactions(config['data']['transactions_train'])
    labels = load_fraud_labels(config['data']['fraud_labels'])
    cards = load_cards_data(config['data']['cards_data'])
    users = load_users_data(config['data']['users_data'])
    mcc_codes = load_mcc_codes(config['data']['mcc_codes'])

    # Merge data
    logger.info("\nMerging all data sources...")
    df = merge_all_data(transactions, labels, cards, users, mcc_codes)

    # Handle missing values
    logger.info("\nHandling missing values...")
    df = handle_missing_values(df)

    # Feature engineering
    logger.info("\n" + "=" * 50)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 50)

    df = create_all_features(df, is_training=True, config=config)

    # Save processed data
    processed_path = config['data']['processed_train']
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info(f"\nProcessed data saved to {processed_path}")

    # Split data temporally
    logger.info("\n" + "=" * 50)
    logger.info("SPLITTING DATA")
    logger.info("=" * 50)

    train_end_date = config['model']['train_end_date']
    train_df, val_df = split_temporal(df, 'timestamp', train_end_date)

    # Prepare features and target
    # Remove non-feature columns
    exclude_cols = ['transaction_id', 'timestamp', 'is_fraud', 'date', 'year_month']
    feature_cols = [col for col in train_df.columns
                   if col not in exclude_cols
                   and not col.endswith('_description')
                   and not col.endswith('_id')]

    # Select only numeric features
    X_train = train_df[feature_cols].select_dtypes(include=[np.number])
    y_train = train_df['is_fraud']
    X_val = val_df[feature_cols].select_dtypes(include=[np.number])
    y_val = val_df['is_fraud']

    logger.info(f"\nTrain set: {X_train.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    logger.info(f"Number of features: {len(X_train.columns)}")

    # Model training
    logger.info("\n" + "=" * 50)
    logger.info("MODEL TRAINING")
    logger.info("=" * 50)

    model_types = config['model']['models_to_train']
    resampling_method = config['model']['resampling_method']
    random_state = config['model']['random_state']

    # Train multiple models
    models = train_multiple_models(
        X_train, y_train,
        X_val, y_val,
        model_types=model_types,
        resampling_method=resampling_method,
        random_state=random_state
    )

    # Select best model
    primary_metric = config['metrics']['primary_metric']
    best_model_name, best_model = select_best_model(models, X_val, y_val, metric=primary_metric)

    # Find optimal threshold
    logger.info("\n" + "=" * 50)
    logger.info("THRESHOLD OPTIMIZATION")
    logger.info("=" * 50)

    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    optimal_threshold, optimal_score = find_optimal_threshold(y_val, y_val_proba, metric='f1')

    # Final evaluation
    logger.info("\n" + "=" * 50)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 50)

    final_metrics = evaluate_model(
        best_model, X_val, y_val,
        threshold=optimal_threshold,
        model_name=f"Final {best_model_name}",
        save_dir='results/figures'
    )

    # Save model
    logger.info("\n" + "=" * 50)
    logger.info("SAVING MODEL")
    logger.info("=" * 50)

    model_path = 'models/fraud_detection_model.pkl'
    best_model.save(model_path)

    # Save metrics and configuration
    results = {
        'best_model': best_model_name,
        'optimal_threshold': float(optimal_threshold),
        'metrics': {k: float(v) for k, v in final_metrics.items()},
        'features': X_train.columns.tolist(),
        'config': config
    }

    save_json(results, 'results/training_results.json')
    logger.info("\nTraining results saved to results/training_results.json")

    # Feature importance
    importance_df = best_model.get_feature_importance()
    if not importance_df.empty:
        importance_df.to_csv('results/feature_importance.csv', index=False)
        logger.info("Feature importance saved to results/feature_importance.csv")

        logger.info("\nTop 20 most important features:")
        print(importance_df.head(20).to_string())

    logger.info("\n" + "=" * 50)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train fraud detection model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()

    try:
        main(args.config)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)
