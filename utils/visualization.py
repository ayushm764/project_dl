"""
Visualization utilities for anomaly detection results.
All plots are saved to the outputs/ directory (no interactive display).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving only
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_reconstruction_errors(train_errors: np.ndarray, test_errors: np.ndarray,
                                test_labels: np.ndarray, threshold: float,
                                output_dir: str):
    """
    Plot reconstruction error distributions for train/test data
    and per-frame test errors colored by ground truth.

    Saves two plots:
      - error_distribution.png  — histogram of train vs test errors
      - frame_errors.png        — per-frame error with threshold line
    """
    os.makedirs(output_dir, exist_ok=True)

    # ─────── Error Distribution Histogram ───────
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=150)
    ax.hist(train_errors, bins=80, alpha=0.6, label='Train (Normal)', color='#2196F3', density=True)
    ax.hist(test_errors, bins=80, alpha=0.6, label='Test (Mixed)', color='#FF5722', density=True)
    ax.axvline(threshold, color='#4CAF50', linewidth=2, linestyle='--',
               label=f'Threshold = {threshold:.6f}')
    ax.set_xlabel('Reconstruction Error (MSE)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'error_distribution.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ─────── Per-Frame Error Plot ───────
    fig, ax = plt.subplots(1, 1, figsize=(16, 5), dpi=150)

    normal_idx = np.where(test_labels == 0)[0]
    anomaly_idx = np.where(test_labels == 1)[0]

    ax.scatter(normal_idx, test_errors[normal_idx], s=4, alpha=0.5,
               color='#2196F3', label='Normal', zorder=2)
    ax.scatter(anomaly_idx, test_errors[anomaly_idx], s=4, alpha=0.5,
               color='#F44336', label='Anomalous', zorder=2)
    ax.axhline(threshold, color='#4CAF50', linewidth=2, linestyle='--',
               label=f'Threshold = {threshold:.6f}', zorder=3)

    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Reconstruction Error (MSE)', fontsize=12)
    ax.set_title('Per-Frame Reconstruction Error (Test Set)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'frame_errors.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def save_metrics_chart(metrics: dict, epoch_losses: list, output_dir: str):
    """
    Save a summary chart with training loss curve and final metrics bar chart.

    Saves: metrics_summary.png
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # ─────── Training Loss Curve ───────
    ax = axes[0]
    epochs = list(range(1, len(epoch_losses) + 1))
    ax.plot(epochs, epoch_losses, linewidth=2, color='#673AB7', marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Average Loss (MSE)', fontsize=12)
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # ─────── Metrics Bar Chart ───────
    ax = axes[1]
    metric_names = ['Precision', 'Recall', 'F1 Score', 'AUC-ROC']
    metric_values = [
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
        metrics['auc'],
    ]
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']
    bars = ax.bar(metric_names, metric_values, color=colors, width=0.5, edgecolor='white')

    # Add value labels on bars
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, 'metrics_summary.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, output_dir: str):
    """
    Plot and save the ROC curve.

    Saves: roc_curve.png
    """
    os.makedirs(output_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7), dpi=150)
    ax.plot(fpr, tpr, color='#E91E63', linewidth=2,
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linewidth=1, linestyle='--', label='Random')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#E91E63')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    plt.tight_layout()
    path = os.path.join(output_dir, 'roc_curve.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
