"""
Preprocessing utilities for UCSD Ped2 dataset.

Loads .tif frame sequences from Train/Test directories and
.bmp ground-truth masks from Test*_gt directories.
"""

import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_frames_from_sequence(sequence_dir: str, image_size: int = 64) -> np.ndarray:
    """
    Load all .tif frames from a single sequence directory.

    Args:
        sequence_dir: Path to a sequence folder (e.g., Train001/)
        image_size:   Target size to resize frames to (square)

    Returns:
        numpy array of shape (N, image_size, image_size) with float32 values in [0, 1]
    """
    tif_files = sorted(glob.glob(os.path.join(sequence_dir, "*.tif")))
    if not tif_files:
        return np.array([], dtype=np.float32)

    frames = []
    for fpath in tif_files:
        img = Image.open(fpath).convert('L')  # Grayscale
        img = img.resize((image_size, image_size), Image.BILINEAR)
        frame = np.array(img, dtype=np.float32) / 255.0
        frames.append(frame)

    return np.stack(frames, axis=0)


def load_gt_masks_from_sequence(gt_dir: str, image_size: int = 64) -> np.ndarray:
    """
    Load all .bmp ground truth masks from a *_gt directory.

    Masks are binarized: pixel > 0 → 1 (anomalous), else 0 (normal).

    Returns:
        numpy array of shape (N, image_size, image_size) with int values {0, 1}
    """
    bmp_files = sorted(glob.glob(os.path.join(gt_dir, "*.bmp")))
    if not bmp_files:
        return np.array([], dtype=np.int32)

    masks = []
    for fpath in bmp_files:
        img = Image.open(fpath).convert('L')
        img = img.resize((image_size, image_size), Image.NEAREST)
        mask = (np.array(img, dtype=np.float32) > 0).astype(np.int32)
        masks.append(mask)

    return np.stack(masks, axis=0)


def load_all_train_frames(data_dir: str, image_size: int = 64) -> np.ndarray:
    """
    Load all training frames from UCSDped2/Train/.

    Args:
        data_dir:   Root UCSDped2 directory
        image_size: Target frame size

    Returns:
        numpy array of shape (total_frames, image_size, image_size)
    """
    train_dir = os.path.join(data_dir, "Train")
    sequence_dirs = sorted([
        os.path.join(train_dir, d) for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d)) and d.startswith("Train")
    ])

    print(f"  Found {len(sequence_dirs)} training sequences")
    all_frames = []

    for seq_dir in tqdm(sequence_dirs, desc="  Loading train sequences"):
        frames = load_frames_from_sequence(seq_dir, image_size)
        if len(frames) > 0:
            all_frames.append(frames)

    all_frames = np.concatenate(all_frames, axis=0)
    print(f"  Total training frames: {all_frames.shape[0]}")
    return all_frames


def load_all_test_data(data_dir: str, image_size: int = 64) -> tuple:
    """
    Load all test frames and corresponding ground truth labels from UCSDped2/Test/.

    For each test sequence, a frame-level label is derived from the pixel-level
    ground-truth mask: if ANY pixel in the mask is anomalous, the frame is labeled 1.

    Args:
        data_dir:   Root UCSDped2 directory
        image_size: Target frame size

    Returns:
        (test_frames, frame_labels)
          test_frames:  numpy array (total_frames, image_size, image_size)
          frame_labels: numpy array (total_frames,) with {0, 1}
    """
    test_dir = os.path.join(data_dir, "Test")

    # Discover test sequence directories (exclude _gt dirs)
    test_seq_names = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
        and d.startswith("Test")
        and "_gt" not in d
    ])

    print(f"  Found {len(test_seq_names)} test sequences")
    all_frames = []
    all_labels = []

    for seq_name in tqdm(test_seq_names, desc="  Loading test sequences"):
        seq_dir = os.path.join(test_dir, seq_name)
        gt_dir = os.path.join(test_dir, f"{seq_name}_gt")

        frames = load_frames_from_sequence(seq_dir, image_size)

        if os.path.isdir(gt_dir):
            masks = load_gt_masks_from_sequence(gt_dir, image_size)
            # Frame-level label: 1 if any pixel is anomalous
            frame_labels = (masks.reshape(masks.shape[0], -1).max(axis=1) > 0).astype(np.int32)
        else:
            # No ground truth → assume all normal
            frame_labels = np.zeros(len(frames), dtype=np.int32)

        # Ensure alignment: use minimum length
        n = min(len(frames), len(frame_labels))
        all_frames.append(frames[:n])
        all_labels.append(frame_labels[:n])

    all_frames = np.concatenate(all_frames, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(f"  Total test frames: {all_frames.shape[0]}")
    print(f"  Anomalous frames:  {all_labels.sum()} / {len(all_labels)} "
          f"({100 * all_labels.mean():.1f}%)")
    return all_frames, all_labels
