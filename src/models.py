"""
Machine learning models for fraud detection
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional, List, Any
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

logger = logging.getLogger(__name__)


class FraudDetectionModel:
    """
    Wrapper class for fraud detection models
    """

    def __init__(self, model_type: str = 'xgboost', random_state: int = 42):
        """
        Initialize model

        Args:
            model_type: Type of model to use
            random_state: Random seed
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def get_model(self, **params) -> Any:
        """
        Get model instance based on model_type

        Args:
            **params: Model parameters

        Returns:
            Model instance
        """
        if self.model_type == 'logistic_regression':
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                **params
            )

        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                **params
            )

        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                random_state=self.random_state,
                **params
            )

        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not installed")
            return xgb.XGBClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                tree_method='hist',
                **params
            )

        elif self.model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not installed")
            return lgb.LGBMClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                **params
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series,
            scale: bool = True,
            **fit_params) -> 'FraudDetectionModel':
        """
        Fit the model

        Args:
            X: Feature matrix
            y: Target vector
            scale: Whether to scale features
            **fit_params: Additional fit parameters

        Returns:
            Self
        """
        logger.info(f"Training {self.model_type} model...")
        logger.info(f"Training set size: {len(X)}")
        logger.info(f"Fraud rate: {y.mean():.4f}")

        self.feature_names = X.columns.tolist()

        # Scale features if needed
        if scale and self.model_type in ['logistic_regression']:
            X_scaled = self.scaler.fit_transform(X)
            X_train = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_train = X

        # Get default model
        if self.model is None:
            self.model = self.get_model()

        # Fit model
        self.model.fit(X_train, y, **fit_params)

        logger.info("Model training completed")
        return self

    def predict(self, X: pd.DataFrame, scale: bool = True) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Feature matrix
            scale: Whether to scale features

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Scale features if needed
        if scale and self.model_type in ['logistic_regression']:
            X_scaled = self.scaler.transform(X)
            X_pred = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_pred = X

        return self.model.predict(X_pred)

    def predict_proba(self, X: pd.DataFrame, scale: bool = True) -> np.ndarray:
        """
        Predict probabilities

        Args:
            X: Feature matrix
            scale: Whether to scale features

        Returns:
            Probability predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Scale features if needed
        if scale and self.model_type in ['logistic_regression']:
            X_scaled = self.scaler.transform(X)
            X_pred = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_pred = X

        return self.model.predict_proba(X_pred)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            logger.warning("Model does not have feature importance")
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return importance_df

    def save(self, filepath: str) -> None:
        """
        Save model to disk

        Args:
            filepath: Path to save model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'FraudDetectionModel':
        """
        Load model from disk

        Args:
            filepath: Path to model file

        Returns:
            FraudDetectionModel instance
        """
        data = joblib.load(filepath)
        instance = cls(model_type=data['model_type'])
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        logger.info(f"Model loaded from {filepath}")
        return instance


def handle_imbalanced_data(X: pd.DataFrame, y: pd.Series,
                           method: str = 'smote',
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handle imbalanced dataset

    Args:
        X: Feature matrix
        y: Target vector
        method: Resampling method ('smote', 'undersample', 'combined', 'none')
        random_state: Random seed

    Returns:
        Resampled X and y
    """
    logger.info(f"Handling imbalanced data using {method}...")
    logger.info(f"Original class distribution: {y.value_counts().to_dict()}")

    if method == 'none' or method == 'class_weight':
        return X, y

    elif method == 'smote':
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)

    elif method == 'undersample':
        rus = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = rus.fit_resample(X, y)

    elif method == 'combined':
        smt = SMOTETomek(random_state=random_state)
        X_resampled, y_resampled = smt.fit_resample(X, y)

    else:
        raise ValueError(f"Unknown resampling method: {method}")

    logger.info(f"Resampled class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")

    # Convert back to DataFrame
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled, name=y.name)

    return X_resampled, y_resampled


def get_hyperparameter_grid(model_type: str) -> Dict[str, List]:
    """
    Get hyperparameter grid for tuning

    Args:
        model_type: Type of model

    Returns:
        Dictionary of hyperparameter ranges
    """
    if model_type == 'logistic_regression':
        return {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': ['balanced', None]
        }

    elif model_type == 'random_forest':
        return {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }

    elif model_type == 'xgboost':
        return {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'scale_pos_weight': [1, 5, 10, 50]
        }

    elif model_type == 'lightgbm':
        return {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, -1],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [31, 63, 127],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'class_weight': ['balanced', None]
        }

    elif model_type == 'gradient_boosting':
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'min_samples_split': [2, 5, 10]
        }

    else:
        return {}


def tune_hyperparameters(model: FraudDetectionModel,
                        X: pd.DataFrame,
                        y: pd.Series,
                        method: str = 'random_search',
                        cv: int = 3,
                        n_iter: int = 50,
                        scoring: str = 'roc_auc') -> FraudDetectionModel:
    """
    Tune model hyperparameters

    Args:
        model: FraudDetectionModel instance
        X: Feature matrix
        y: Target vector
        method: Search method ('random_search' or 'grid_search')
        cv: Number of cross-validation folds
        n_iter: Number of iterations for random search
        scoring: Scoring metric

    Returns:
        Tuned model
    """
    logger.info(f"Tuning hyperparameters using {method}...")

    param_grid = get_hyperparameter_grid(model.model_type)

    if not param_grid:
        logger.warning(f"No hyperparameter grid defined for {model.model_type}")
        return model

    base_model = model.get_model()

    if method == 'random_search':
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=model.random_state,
            verbose=1
        )
    elif method == 'grid_search':
        search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
    else:
        raise ValueError(f"Unknown search method: {method}")

    # Fit search
    search.fit(X, y)

    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best {scoring} score: {search.best_score_:.4f}")

    # Update model with best estimator
    model.model = search.best_estimator_

    return model


def train_multiple_models(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         X_val: pd.DataFrame,
                         y_val: pd.Series,
                         model_types: List[str],
                         resampling_method: str = 'smote',
                         random_state: int = 42) -> Dict[str, FraudDetectionModel]:
    """
    Train multiple models and compare

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        model_types: List of model types to train
        resampling_method: Method to handle imbalanced data
        random_state: Random seed

    Returns:
        Dictionary of trained models
    """
    from .evaluation import evaluate_model  # Import here to avoid circular import

    logger.info(f"Training {len(model_types)} different models...")

    models = {}

    for model_type in model_types:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_type}")
        logger.info(f"{'='*50}")

        try:
            # Handle imbalanced data
            X_train_resampled, y_train_resampled = handle_imbalanced_data(
                X_train, y_train, method=resampling_method, random_state=random_state
            )

            # Train model
            model = FraudDetectionModel(model_type=model_type, random_state=random_state)
            model.fit(X_train_resampled, y_train_resampled)

            # Evaluate on validation set
            metrics = evaluate_model(model, X_val, y_val, model_name=model_type)

            models[model_type] = model

        except Exception as e:
            logger.error(f"Error training {model_type}: {str(e)}")
            continue

    return models


def select_best_model(models: Dict[str, FraudDetectionModel],
                     X_val: pd.DataFrame,
                     y_val: pd.Series,
                     metric: str = 'pr_auc') -> Tuple[str, FraudDetectionModel]:
    """
    Select best model based on validation metric

    Args:
        models: Dictionary of trained models
        X_val: Validation features
        y_val: Validation target
        metric: Metric to use for selection

    Returns:
        Tuple of (best_model_name, best_model)
    """
    from .evaluation import calculate_metrics  # Import here to avoid circular import

    logger.info(f"Selecting best model based on {metric}...")

    best_score = -np.inf
    best_model_name = None
    best_model = None

    for model_name, model in models.items():
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        metrics = calculate_metrics(y_val, y_pred_proba)

        score = metrics.get(metric, 0)
        logger.info(f"{model_name}: {metric} = {score:.4f}")

        if score > best_score:
            best_score = score
            best_model_name = model_name
            best_model = model

    logger.info(f"\nBest model: {best_model_name} with {metric} = {best_score:.4f}")

    return best_model_name, best_model
