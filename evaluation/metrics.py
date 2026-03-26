"""
Evaluation metrics for anomaly detection.
Computes frame-level precision, recall, F1, AUC, and confusion matrix.
"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_scores: np.ndarray = None) -> dict:
    """
    Compute evaluation metrics.

    Args:
        y_true:   Ground-truth labels (0 = normal, 1 = anomalous)
        y_pred:   Predicted labels (0/1)
        y_scores: Continuous anomaly scores (reconstruction errors) for AUC

    Returns:
        Dictionary with precision, recall, f1, auc, confusion_matrix
    """
    metrics = {}

    metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))

    if y_scores is not None and len(np.unique(y_true)) > 1:
        metrics['auc'] = float(roc_auc_score(y_true, y_scores))
    else:
        metrics['auc'] = 0.0

    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=['Normal', 'Anomalous'], zero_division=0
    )

    return metrics


def print_metrics(metrics: dict):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Precision : {metrics['precision']:.4f}  ({metrics['precision']*100:.1f}%)")
    print(f"  Recall    : {metrics['recall']:.4f}  ({metrics['recall']*100:.1f}%)")
    print(f"  F1 Score  : {metrics['f1']:.4f}  ({metrics['f1']*100:.1f}%)")
    print(f"  AUC-ROC   : {metrics['auc']:.4f}")
    print(f"{'='*60}")
    print(f"\n  Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"                  Predicted")
    print(f"                  Normal  Anomalous")
    print(f"  Actual Normal   {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"  Actual Anomalous{cm[1][0]:6d}  {cm[1][1]:6d}")
    print(f"\n{metrics['classification_report']}")
