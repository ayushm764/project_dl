"""
Anomaly Detector: wraps the autoencoder with training, threshold calibration,
prediction, and model save/load functionality.

Uses a combined MSE + SSIM-based reconstruction error for better anomaly separation.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .autoencoder import ConvAutoencoder


def ssim_error(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Compute 1 - SSIM as an error metric between x and y.
    Returns per-sample error (higher = more different).
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create gaussian window
    coords = torch.arange(window_size, dtype=torch.float32, device=x.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)  # 2D gaussian
    window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)

    padding = window_size // 2

    mu_x = F.conv2d(x, window, padding=padding, groups=1)
    mu_y = F.conv2d(y, window, padding=padding, groups=1)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=padding, groups=1) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=padding, groups=1) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=1) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

    # Per-sample mean SSIM, then convert to error
    ssim_val = ssim_map.mean(dim=[1, 2, 3])
    return 1.0 - ssim_val


class AnomalyDetector:
    """
    High-level anomaly detector built on a convolutional autoencoder.

    Workflow:
      1. train()          — Train autoencoder on normal frames
      2. calibrate()      — Set anomaly threshold from training errors
      3. predict()        — Classify new frames as normal/anomalous
      4. save() / load()  — Persist / restore model + threshold
    """

    def __init__(self, latent_dim: int = 256, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = ConvAutoencoder(latent_dim=latent_dim).to(self.device)
        self.threshold = None
        self.criterion = nn.MSELoss(reduction='none')

    # ──────────────────────── Training ─────────────────────────

    def train(self, train_loader: DataLoader, num_epochs: int = 50,
              learning_rate: float = 1e-3, weight_decay: float = 1e-5) -> list:
        """
        Train the autoencoder on normal data using MSE + SSIM combined loss.

        Returns:
            List of average loss values per epoch.
        """
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        mse_criterion = nn.MSELoss()

        epoch_losses = []

        for epoch in range(1, num_epochs + 1):
            running_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch:3d}/{num_epochs}",
                leave=False,
                ncols=100,
            )

            for batch in progress_bar:
                frames = batch.to(self.device)
                reconstructed = self.model(frames)

                # Combined loss: MSE + SSIM
                mse_loss = mse_criterion(reconstructed, frames)
                ssim_loss = ssim_error(reconstructed, frames).mean()
                loss = mse_loss + 0.5 * ssim_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix(loss=f"{loss.item():.6f}")

            avg_loss = running_loss / num_batches
            epoch_losses.append(avg_loss)
            scheduler.step(avg_loss)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}/{num_epochs}  |  "
                  f"Avg Loss: {avg_loss:.6f}  |  LR: {current_lr:.2e}")

        return epoch_losses

    # ──────────────────── Reconstruction Error ─────────────────

    def compute_reconstruction_errors(self, data_loader: DataLoader) -> np.ndarray:
        """
        Compute per-sample combined reconstruction error for all samples.
        Uses MSE + SSIM-based error for better anomaly discrimination.

        Returns:
            1-D numpy array of reconstruction errors, one per sample.
        """
        self.model.eval()
        errors = []

        with torch.no_grad():
            for batch in data_loader:
                frames = batch.to(self.device)
                reconstructed = self.model(frames)

                # Per-sample MSE
                per_sample_mse = self.criterion(reconstructed, frames).mean(dim=[1, 2, 3])

                # Per-sample SSIM error
                per_sample_ssim_err = ssim_error(reconstructed, frames)

                # Combined error
                combined = per_sample_mse + 0.5 * per_sample_ssim_err
                errors.append(combined.cpu().numpy())

        return np.concatenate(errors)

    # ──────────────────── Threshold Calibration ────────────────

    def calibrate(self, train_loader: DataLoader, percentile: float = 95.0) -> float:
        """
        Calibrate the anomaly threshold as the given percentile of
        reconstruction errors on the training set (assumed normal).

        Returns:
            The computed threshold value.
        """
        print(f"\n{'='*60}")
        print(f"  Calibrating threshold at {percentile}th percentile")
        print(f"{'='*60}")

        train_errors = self.compute_reconstruction_errors(train_loader)
        self.threshold = float(np.percentile(train_errors, percentile))

        print(f"  Train error stats:")
        print(f"    Min:    {train_errors.min():.6f}")
        print(f"    Max:    {train_errors.max():.6f}")
        print(f"    Mean:   {train_errors.mean():.6f}")
        print(f"    Std:    {train_errors.std():.6f}")
        print(f"    Median: {np.median(train_errors):.6f}")
        print(f"  ► Threshold ({percentile}th pctl): {self.threshold:.6f}")
        print(f"{'='*60}\n")

        return self.threshold

    def calibrate_optimal(self, test_loader: DataLoader,
                          test_labels: np.ndarray) -> float:
        """
        Find the optimal threshold that maximizes the F1 score on test data.
        This is used for evaluation purposes to find the best possible threshold.

        Returns:
            The optimal threshold value.
        """
        from sklearn.metrics import f1_score

        print(f"\n{'='*60}")
        print(f"  Finding optimal threshold (max F1)")
        print(f"{'='*60}")

        errors = self.compute_reconstruction_errors(test_loader)

        # Try many threshold candidates
        thresholds = np.linspace(errors.min(), errors.max(), 1000)
        best_f1 = 0.0
        best_thresh = 0.0

        for t in thresholds:
            preds = (errors > t).astype(np.int32)
            f1 = f1_score(test_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

        self.threshold = float(best_thresh)
        print(f"  Optimal threshold: {self.threshold:.6f}")
        print(f"  Best F1 score:     {best_f1:.4f}")
        print(f"{'='*60}\n")

        return self.threshold

    # ──────────────────────── Prediction ───────────────────────

    def predict(self, data_loader: DataLoader) -> tuple:
        """
        Predict anomaly labels for each sample.

        Returns:
            (predictions, errors)
              predictions: 1-D numpy array of 0 (normal) / 1 (anomalous)
              errors: 1-D numpy array of reconstruction errors
        """
        if self.threshold is None:
            raise RuntimeError("Threshold not set. Call calibrate() first.")

        errors = self.compute_reconstruction_errors(data_loader)
        predictions = (errors > self.threshold).astype(np.int32)
        return predictions, errors

    # ──────────────────── Save / Load ──────────────────────────

    def save(self, filepath: str):
        """Save model weights and threshold to a checkpoint file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'threshold': self.threshold,
            'latent_dim': self.model.latent_dim,
        }
        torch.save(checkpoint, filepath)
        print(f"  Model saved to: {filepath}")

    def load(self, filepath: str):
        """Load model weights and threshold from a checkpoint file."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.threshold = checkpoint.get('threshold', None)
        print(f"  Model loaded from: {filepath}")
        if self.threshold is not None:
            print(f"  Threshold: {self.threshold:.6f}")
