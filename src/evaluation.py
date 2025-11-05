"""
Model evaluation and metrics calculation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Tuple, Optional, Any
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray,
                     y_pred_proba: np.ndarray,
                     threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate all evaluation metrics

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba),
        'threshold': threshold
    }

    return metrics


def find_optimal_threshold(y_true: np.ndarray,
                          y_pred_proba: np.ndarray,
                          metric: str = 'f1') -> Tuple[float, float]:
    """
    Find optimal classification threshold

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall')

    Returns:
        Tuple of (optimal_threshold, optimal_score)
    """
    logger.info(f"Finding optimal threshold based on {metric}...")

    thresholds = np.linspace(0.1, 0.9, 100)
    scores = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        scores.append(score)

    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_score = scores[optimal_idx]

    logger.info(f"Optimal threshold: {optimal_threshold:.3f} with {metric} = {optimal_score:.4f}")

    return optimal_threshold, optimal_score


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save plot (optional)
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Fraud', 'Fraud'],
                yticklabels=['Not Fraud', 'Fraud'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")

    plt.close()


def plot_roc_curve(y_true: np.ndarray,
                  y_pred_proba: np.ndarray,
                  save_path: Optional[str] = None,
                  figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot ROC curve

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save plot (optional)
        figsize: Figure size
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")

    plt.close()


def plot_precision_recall_curve(y_true: np.ndarray,
                                y_pred_proba: np.ndarray,
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Plot Precision-Recall curve

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save plot (optional)
        figsize: Figure size
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)

    plt.figure(figsize=figsize)
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"PR curve saved to {save_path}")

    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame,
                           top_n: int = 20,
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot feature importance

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to plot
        save_path: Path to save plot (optional)
        figsize: Figure size
    """
    top_features = importance_df.head(top_n)

    plt.figure(figsize=figsize)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")

    plt.close()


def evaluate_model(model: Any,
                  X: pd.DataFrame,
                  y: pd.Series,
                  threshold: float = 0.5,
                  model_name: str = "Model",
                  save_dir: Optional[str] = None) -> Dict[str, float]:
    """
    Comprehensive model evaluation

    Args:
        model: Trained model
        X: Feature matrix
        y: True labels
        threshold: Classification threshold
        model_name: Name of the model
        save_dir: Directory to save plots (optional)

    Returns:
        Dictionary of metrics
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluating {model_name}")
    logger.info(f"{'='*50}")

    # Get predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate metrics
    metrics = calculate_metrics(y, y_pred_proba, threshold)

    # Print metrics
    logger.info(f"\nMetrics (threshold = {threshold}):")
    for metric, value in metrics.items():
        if metric != 'threshold':
            logger.info(f"  {metric}: {value:.4f}")

    # Print classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y, y_pred, target_names=['Not Fraud', 'Fraud']))

    # Print confusion matrix
    cm = confusion_matrix(y, y_pred)
    logger.info("\nConfusion Matrix:")
    logger.info(f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    logger.info(f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}")

    # Save plots if save_dir provided
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        model_slug = model_name.lower().replace(' ', '_')

        plot_confusion_matrix(y, y_pred,
                            save_path=f"{save_dir}/{model_slug}_confusion_matrix.png")
        plot_roc_curve(y, y_pred_proba,
                      save_path=f"{save_dir}/{model_slug}_roc_curve.png")
        plot_precision_recall_curve(y, y_pred_proba,
                                   save_path=f"{save_dir}/{model_slug}_pr_curve.png")

        # Feature importance
        if hasattr(model, 'get_feature_importance'):
            importance_df = model.get_feature_importance()
            if not importance_df.empty:
                plot_feature_importance(importance_df,
                                      save_path=f"{save_dir}/{model_slug}_feature_importance.png")

    return metrics


def compare_models(models_metrics: Dict[str, Dict[str, float]],
                  save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Compare multiple models

    Args:
        models_metrics: Dictionary of {model_name: metrics_dict}
        save_path: Path to save comparison plot (optional)

    Returns:
        DataFrame with comparison
    """
    logger.info("Comparing models...")

    comparison_df = pd.DataFrame(models_metrics).T
    comparison_df = comparison_df.sort_values('pr_auc', ascending=False)

    logger.info("\nModel Comparison:")
    logger.info(comparison_df.to_string())

    if save_path:
        # Plot comparison
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            comparison_df[metric].plot(kind='barh', ax=ax)
            ax.set_xlabel(metric.upper())
            ax.set_title(f'{metric.upper()} Comparison')

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison saved to {save_path}")
        plt.close()

    return comparison_df


def calculate_business_impact(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             transaction_amounts: np.ndarray,
                             cost_per_fp: float = 10.0) -> Dict[str, float]:
    """
    Calculate business impact of the model

    Args:
        y_true: True labels
        y_pred: Predicted labels
        transaction_amounts: Transaction amounts
        cost_per_fp: Cost per false positive (e.g., manual review)

    Returns:
        Dictionary with business metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Amount of fraud detected
    fraud_detected_amount = transaction_amounts[(y_true == 1) & (y_pred == 1)].sum()

    # Amount of fraud missed
    fraud_missed_amount = transaction_amounts[(y_true == 1) & (y_pred == 0)].sum()

    # Cost of false positives
    false_positive_cost = fp * cost_per_fp

    # Net benefit
    net_benefit = fraud_detected_amount - false_positive_cost

    metrics = {
        'fraud_detected_count': tp,
        'fraud_missed_count': fn,
        'fraud_detected_amount': fraud_detected_amount,
        'fraud_missed_amount': fraud_missed_amount,
        'fraud_detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'false_positive_count': fp,
        'false_positive_cost': false_positive_cost,
        'net_benefit': net_benefit,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0
    }

    logger.info("\nBusiness Impact:")
    logger.info(f"  Fraud detected: {tp} transactions (${fraud_detected_amount:,.2f})")
    logger.info(f"  Fraud missed: {fn} transactions (${fraud_missed_amount:,.2f})")
    logger.info(f"  False positives: {fp} transactions (cost: ${false_positive_cost:,.2f})")
    logger.info(f"  Net benefit: ${net_benefit:,.2f}")
    logger.info(f"  Fraud detection rate: {metrics['fraud_detection_rate']:.2%}")

    return metrics


def analyze_errors(X: pd.DataFrame,
                  y_true: np.ndarray,
                  y_pred: np.ndarray,
                  y_pred_proba: np.ndarray,
                  top_n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze false positives and false negatives

    Args:
        X: Feature matrix
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        top_n: Number of examples to return

    Returns:
        Tuple of (false_positives_df, false_negatives_df)
    """
    logger.info("Analyzing errors...")

    # False positives
    fp_mask = (y_true == 0) & (y_pred == 1)
    fp_df = X[fp_mask].copy()
    fp_df['true_label'] = y_true[fp_mask]
    fp_df['predicted_proba'] = y_pred_proba[fp_mask]
    fp_df = fp_df.sort_values('predicted_proba', ascending=False).head(top_n)

    # False negatives
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_df = X[fn_mask].copy()
    fn_df['true_label'] = y_true[fn_mask]
    fn_df['predicted_proba'] = y_pred_proba[fn_mask]
    fn_df = fn_df.sort_values('predicted_proba', ascending=True).head(top_n)

    logger.info(f"\nFalse Positives: {fp_mask.sum()}")
    logger.info(f"False Negatives: {fn_mask.sum()}")

    return fp_df, fn_df
