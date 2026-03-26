"""
PyTorch Dataset for UCSD Ped2 frames.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class UCSDPed2Dataset(Dataset):
    """
    PyTorch Dataset wrapping pre-loaded grayscale frames.

    Each sample is a (1, H, W) float32 tensor in [0, 1].
    """

    def __init__(self, frames: np.ndarray):
        """
        Args:
            frames: numpy array of shape (N, H, W) with float32 values in [0, 1]
        """
        super().__init__()
        self.frames = frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        frame = self.frames[idx]
        # Add channel dimension: (H, W) → (1, H, W)
        tensor = torch.from_numpy(frame).unsqueeze(0).float()
        return tensor
