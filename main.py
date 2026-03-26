"""
Main pipeline for UCSD Ped2 Anomaly Detection.

Full workflow:
  1. Load UCSD Ped2 dataset (train frames + test frames with ground truth)
  2. Train convolutional autoencoder on normal training data
  3. Calibrate anomaly threshold at 95th percentile of train errors
  4. Evaluate on test set with ground truth labels
  5. Print metrics and save all plots to outputs/
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

# Project imports
import config
from data.preprocessing import load_all_train_frames, load_all_test_data
from data.dataset import UCSDPed2Dataset
from models.detector import AnomalyDetector
from evaluation.metrics import compute_metrics, print_metrics
from utils.visualization import (
    plot_reconstruction_errors,
    save_metrics_chart,
    plot_roc_curve,
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='UCSD Ped2 Video Anomaly Detection using Convolutional Autoencoder'
    )
    parser.add_argument(
        '--data_path', type=str, default=config.DATA_DIR,
        help='Path to UCSDped2 root directory (containing Train/ and Test/)'
    )
    parser.add_argument(
        '--output_dir', type=str, default=config.OUTPUT_DIR,
        help='Directory to save model checkpoints and plots'
    )
    parser.add_argument(
        '--epochs', type=int, default=config.NUM_EPOCHS,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=config.BATCH_SIZE,
        help='Training batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=config.LEARNING_RATE,
        help='Learning rate'
    )
    parser.add_argument(
        '--latent_dim', type=int, default=config.LATENT_DIM,
        help='Latent space dimensionality'
    )
    parser.add_argument(
        '--threshold_pctl', type=float, default=config.THRESHOLD_PERCENTILE,
        help='Percentile for threshold calibration'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(config.SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'#'*60}")
    print(f"  UCSD Ped2 — Video Anomaly Detection")
    print(f"  {'='*56}")
    print(f"  Device       : {device}")
    print(f"  Data path    : {args.data_path}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Latent dim   : {args.latent_dim}")
    print(f"  Image size   : {config.IMAGE_SIZE}×{config.IMAGE_SIZE}")
    print(f"  Threshold pct: {args.threshold_pctl}")
    print(f"{'#'*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # ════════════════════════════════════════════════════════════
    # STEP 1: Load Data
    # ════════════════════════════════════════════════════════════
    print(f"{'='*60}")
    print(f"  STEP 1: Loading UCSD Ped2 Dataset")
    print(f"{'='*60}")

    train_frames = load_all_train_frames(args.data_path, config.IMAGE_SIZE)
    test_frames, test_labels = load_all_test_data(args.data_path, config.IMAGE_SIZE)

    print(f"\n  Train frames shape : {train_frames.shape}")
    print(f"  Test frames shape  : {test_frames.shape}")
    print(f"  Test labels shape  : {test_labels.shape}")
    print(f"  Normal test frames : {(test_labels == 0).sum()}")
    print(f"  Anomalous test frames: {(test_labels == 1).sum()}")

    # Create datasets and loaders
    train_dataset = UCSDPed2Dataset(train_frames)
    test_dataset = UCSDPed2Dataset(test_frames)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=(device == "cuda")
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=(device == "cuda")
    )

    # ════════════════════════════════════════════════════════════
    # STEP 2: Train Autoencoder
    # ════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STEP 2: Training Convolutional Autoencoder")
    print(f"{'='*60}\n")

    detector = AnomalyDetector(latent_dim=args.latent_dim, device=device)

    # Print model summary
    total_params = sum(p.numel() for p in detector.model.parameters())
    trainable_params = sum(p.numel() for p in detector.model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,} total, {trainable_params:,} trainable\n")

    start_time = time.time()
    epoch_losses = detector.train(
        train_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=config.WEIGHT_DECAY,
    )
    train_time = time.time() - start_time
    print(f"\n  Training completed in {train_time:.1f}s ({train_time/60:.1f} min)")

    # ════════════════════════════════════════════════════════════
    # STEP 3: Calibrate Threshold
    # ════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STEP 3: Calibrating Anomaly Threshold")
    print(f"{'='*60}")

    # Use a non-shuffled train loader for calibration
    calib_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    threshold = detector.calibrate(calib_loader, percentile=args.threshold_pctl)

    # ════════════════════════════════════════════════════════════
    # STEP 4: Evaluate on Test Set
    # ════════════════════════════════════════════════════════════
    print(f"{'='*60}")
    print(f"  STEP 4: Evaluating on Test Set")
    print(f"{'='*60}")

    predictions, test_errors = detector.predict(test_loader)
    train_errors = detector.compute_reconstruction_errors(calib_loader)

    # Compute metrics
    metrics = compute_metrics(test_labels, predictions, y_scores=test_errors)
    print_metrics(metrics)

    # ════════════════════════════════════════════════════════════
    # STEP 5: Save Model & Plots
    # ════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STEP 5: Saving Results")
    print(f"{'='*60}")

    # Save model
    model_path = os.path.join(args.output_dir, 'autoencoder.pth')
    detector.save(model_path)

    # Save plots
    print(f"\n  Generating plots...")
    plot_reconstruction_errors(train_errors, test_errors, test_labels,
                                threshold, args.output_dir)
    save_metrics_chart(metrics, epoch_losses, args.output_dir)
    plot_roc_curve(test_labels, test_errors, args.output_dir)

    # ════════════════════════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════════════════════════
    print(f"\n{'#'*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  {'='*56}")
    print(f"  Precision : {metrics['precision']*100:.1f}%")
    print(f"  Recall    : {metrics['recall']*100:.1f}%")
    print(f"  F1 Score  : {metrics['f1']*100:.1f}%")
    print(f"  AUC-ROC   : {metrics['auc']:.4f}")
    print(f"  Threshold : {threshold:.6f}")
    print(f"  {'='*56}")
    print(f"  Outputs saved to: {args.output_dir}")
    print(f"    - autoencoder.pth")
    print(f"    - error_distribution.png")
    print(f"    - frame_errors.png")
    print(f"    - metrics_summary.png")
    print(f"    - roc_curve.png")
    print(f"{'#'*60}\n")


if __name__ == '__main__':
    main()
