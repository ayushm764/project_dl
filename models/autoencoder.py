"""
Convolutional Autoencoder for frame-level anomaly detection.

Architecture:
  Encoder: 1×64×64 → Conv layers → Flatten → FC → 256-dim latent
  Decoder: 256-dim latent → FC → Reshape → ConvTranspose layers → 1×64×64

The encoder uses strided convolutions (no pooling) for downsampling.
The decoder uses transposed convolutions for upsampling.
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """Symmetric convolutional autoencoder for grayscale 64×64 frames."""

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        # ───────────────────── Encoder ─────────────────────
        # Input: (B, 1, 64, 64)
        self.encoder_conv = nn.Sequential(
            # Layer 1: 1×64×64  → 32×32×32
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 32×32×32 → 64×16×16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 64×16×16 → 128×8×8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 128×8×8 → 256×4×4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Flatten 256×4×4 = 4096 → latent_dim
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ───────────────────── Decoder ─────────────────────
        # latent_dim → 256×4×4
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Upsample back to 1×64×64
        self.decoder_conv = nn.Sequential(
            # Layer 1: 256×4×4 → 128×8×8
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 128×8×8 → 64×16×16
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 64×16×16 → 32×32×32
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 32×32×32 → 1×64×64
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        h = self.encoder_conv(x)
        z = self.encoder_fc(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        h = self.decoder_fc(z)
        h = h.view(-1, 256, 4, 4)
        x_hat = self.decoder_conv(h)
        return x_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode then decode."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
