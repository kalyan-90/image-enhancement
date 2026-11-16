"""
Configuration file for Retinex Low-Light Image Enhancement Project.
Defines paths, hyperparameters, training configurations, and loss weights.
"""

import os
import torch

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "LOL")
TRAIN_LOW_DIR = os.path.join(DATASET_DIR, "train", "low")
TRAIN_HIGH_DIR = os.path.join(DATASET_DIR, "train", "high")
TEST_LOW_DIR = os.path.join(DATASET_DIR, "test", "low")
TEST_HIGH_DIR = os.path.join(DATASET_DIR, "test", "high")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
CORRECTED_BASELINE_DIR = os.path.join(CHECKPOINT_DIR, "corrected_baseline")
IMPROVED_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "improved")
ABLATION_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "ablations")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
SPLIT_CONFIG_PATH = os.path.join(BASE_DIR, "configs", "train_val_split.json")

# Ensure necessary output directories exist
for path in [IMPROVED_CHECKPOINT_DIR, ABLATION_CHECKPOINT_DIR, RESULTS_DIR, EXPERIMENTS_DIR]:
    os.makedirs(path, exist_ok=True)

# Training Hyperparameters
RANDOM_SEED = 42
BATCH_SIZE = 4
PATCH_SIZE = 48
NUM_WORKERS = 0
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Epoch Settings
EPOCHS_DECOM = 20
EPOCHS_RELIGHT = 20
EPOCHS_IGNRR = 25

# Loss Function Weights for Proposed IGNRR
LAMBDA_L1 = 1.0
LAMBDA_SSIM = 0.2
LAMBDA_COLOR = 0.5     # Cosine angular color loss to minimize Delta E 2000
LAMBDA_EDGE = 0.1      # Sobel edge gradient loss
LAMBDA_SMOOTH = 0.1    # Illumination smoothness loss

# Hardware Detection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
