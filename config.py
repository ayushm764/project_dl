"""
Configuration file for UCSD Ped2 Anomaly Detection.
All hyperparameters, paths, and settings are centralized here.
"""

import os

# ─────────────────────────── Paths ─────────────────────────────
# Base project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset root — the UCSDped2 folder containing Train/ and Test/
DATA_DIR = os.path.join(PROJECT_DIR, "UCSDped2")

# Output directory for saved models, plots, metrics
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model checkpoint path
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "autoencoder.pth")

# ─────────────────────────── Image ─────────────────────────────
IMAGE_SIZE = 64          # Resize frames to IMAGE_SIZE x IMAGE_SIZE
IMAGE_CHANNELS = 1       # Grayscale

# ─────────────────────────── Model ─────────────────────────────
LATENT_DIM = 256         # Bottleneck dimensionality

# ─────────────────────── Training ──────────────────────────────
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0          # Windows-safe: no multiprocessing workers
SEED = 42

# ─────────────────── Anomaly Detection ─────────────────────────
THRESHOLD_PERCENTILE = 95   # Calibrate threshold at this percentile of train errors

# ─────────────────── Visualization ─────────────────────────────
PLOT_DPI = 150
PLOT_FIGSIZE = (12, 6)
