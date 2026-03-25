"""
Training script for the deskewing model with UI integration
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import cv2
import json
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import logging

# Now import from src
from src.deskewing import DeskewModel
from config import TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, TRAIN_LIST_PATH, MODELS_DIR

def load_training_data(progress_callback=None):
    """Load training data from the dataset"""
    print("Loading training data...")
    
    images = []
    angles = []
    
    # Load training list
    with open(TRAIN_LIST_PATH, 'r') as f:
        train_list = json.load(f)
    
    # Use all available training images
    total_images = len(train_list)
    
    for idx, img_name in enumerate(train_list[:total_images]):
        img_path = TRAIN_IMAGES_DIR / img_name
        label_path = TRAIN_LABELS_DIR / f"{img_name.replace('.png', '.txt')}"
        
        if img_path.exists() and label_path.exists():
            # Load and preprocess image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # Resize and normalize
            img = cv2.resize(img, (224, 224))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
            
            # Load angle
            with open(label_path, 'r') as f:
                angle = float(f.read().strip())
            
            images.append(img)
            angles.append(angle)
            
            # Report progress
            if progress_callback:
                progress = (idx + 1) / total_images
                progress_callback(progress, f"Loading image {idx + 1}/{total_images}")
            
            if (idx + 1) % 10 == 0:
                print(f"Loaded {idx + 1}/{total_images} images...")
    
    if len(images) == 0:
        return np.array([]), np.array([])
    
    print(f"Successfully loaded {len(images)} images")
    return np.array(images), np.array(angles)

def train_model(progress_callback=None):
    """Train the deskewing model"""
    print("Starting model training...")
    
    # Load data
    if progress_callback:
        progress_callback(0.1, "Loading training data...")
    
    X, y = load_training_data(progress_callback=lambda p, msg: progress_callback(0.1 + p * 0.3, msg) if progress_callback else None)
    
    if len(X) == 0:
        print("No training data found!")
        return False
    
    print(f"Loaded {len(X)} training samples")
    
    if progress_callback:
        progress_callback(0.4, f"Loaded {len(X)} images. Preparing model...")
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Build model
    if progress_callback:
        progress_callback(0.5, "Building CNN model...")
    
    model = DeskewModel()
    model.compile_model()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(
            str(MODELS_DIR / 'deskew_model_best.weights.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        )
    ]
    
    # Train model
    if progress_callback:
        progress_callback(0.6, "Training model (this may take a few minutes)...")
    
    print("Starting training...")
    history = model.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    if progress_callback:
        progress_callback(0.95, "Saving model...")
    
    model.model.save_weights(str(MODELS_DIR / 'deskew_model_final.weights.h5'))
    print("Model training completed!")
    
    if progress_callback:
        progress_callback(1.0, "Training completed!")
    
    return True

if __name__ == "__main__":
    train_model()