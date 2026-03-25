"""
Configuration file for Scanned Document Restoration Pipeline
Based on Cosmas Kiptoo Sang's Research Project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Data paths
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
TRAIN_IMAGES_DIR = IMAGES_DIR / "train"
TEST_IMAGES_DIR = IMAGES_DIR / "test"
TRAIN_LABELS_DIR = LABELS_DIR / "train"

# File lists
TRAIN_LIST_PATH = DATA_DIR / "train_list.json"
TEST_LIST_PATH = DATA_DIR / "test_list.json"

# Model paths
DESKEW_MODEL_PATH = MODELS_DIR / "deskew_cnn_model.h5"
DESKEW_MODEL_WEIGHTS_PATH = MODELS_DIR / "deskew_cnn_weights.h5"

# Results paths
DENOISING_RESULTS_PATH = RESULTS_DIR / "denoising_results.csv"
DESKEWING_RESULTS_PATH = RESULTS_DIR / "deskewing_results.csv"
OCR_RESULTS_PATH = RESULTS_DIR / "ocr_results.csv"
PIPELINE_RESULTS_PATH = RESULTS_DIR / "pipeline_results.csv"

# Image processing parameters
IMAGE_SIZE = (224, 224)  # CNN input size
IMAGE_CHANNELS = 1  # Grayscale

# Denoising parameters
DENOISING_METHODS = {
    "gaussian_blur": {"ksize": (5, 5)},
    "median_filter": {"ksize": 5},
    "bilateral_filter": {"d": 9, "sigmaColor": 75, "sigmaSpace": 75},
    "non_local_means": {"h": 10, "templateWindowSize": 7, "searchWindowSize": 21},
    "morphological_opening": {"kernel_size": 3},
    "clahe": {"clip_limit": 3.0, "tile_grid_size": (8, 8)},
}

# CNN model parameters
CNN_PARAMS = {
    "input_shape": (224, 224, 1),
    "conv_filters": [32, 64, 128, 256],
    "dense_units": 128,
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "batch_size": 16,
    "epochs": 50,
    "validation_split": 0.2,
}

# Training parameters
TRAIN_SPLIT = 0.8  # 80% training, 20% validation
RANDOM_SEED = 42

# OCR parameters
OCR_CONFIG = {
    "lang": "eng",
    "config": "--psm 6 --oem 1",  # PSM 6: Uniform block of text, OEM 1: LSTM only (better accuracy)
    "timeout": 30,  # seconds
}

# Evaluation metrics thresholds
PSNR_THRESHOLD = 30.0  # dB
SSIM_THRESHOLD = 0.9
ANGLE_MAE_THRESHOLD = 2.0  # degrees

# Ground truth text (from research document)
GROUND_TRUTH_TEXT = """Surety
This document certifies that the undersigned guarantees
the performance and obligations of the principal party
in accordance with the terms and conditions specified
in the agreement dated as referenced herein.
The surety bond remains in full force and effect until
such time as all obligations have been fulfilled or
the bond is formally released by the obligee.
In witness whereof, the surety has executed this bond
on the date first above written."""

# Create directories
for directory in [
    DATA_DIR, IMAGES_DIR, LABELS_DIR, TRAIN_IMAGES_DIR, TEST_IMAGES_DIR, 
    TRAIN_LABELS_DIR, MODELS_DIR, RESULTS_DIR, SCRIPTS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)