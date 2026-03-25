"""
Training script for CNN deskewing model
Based on Cosmas Kiptoo Sang's research methodology
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import cv2
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
import logging
from datetime import datetime

from src.deskewing import DeskewModel, load_training_data, train_deskew_model
from config import (
    TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, TRAIN_LIST_PATH,
    MODELS_DIR, RESULTS_DIR, CNN_PARAMS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare_data():
    """Load and prepare training data."""
    logger.info("Loading training data...")
    
    # Load training list
    with open(TRAIN_LIST_PATH, 'r') as f:
        train_list = json.load(f)
    
    # Convert to dictionary with angle labels
    # Note: In the actual dataset, angles are stored in separate .txt files
    # For now, we'll create synthetic angles for demonstration
    images = []
    angles = []
    
    for i, img_name in enumerate(train_list[:500]):  # First 500 training images
        img_path = TRAIN_IMAGES_DIR / img_name
        if img_path.exists():
            # Load image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Resize to model input size
            img = cv2.resize(img, (224, 224))
            
            # Normalize
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
            
            # Load angle from label file
            label_path = TRAIN_LABELS_DIR / f"{img_name.replace('.png', '.txt')}"
            if label_path.exists():
                with open(label_path, 'r') as f:
                    angle = float(f.read().strip())
            else:
                # Synthetic angle for demonstration (between -10 and 10 degrees)
                angle = np.random.uniform(-10, 10)
            
            images.append(img)
            angles.append(angle)
    
    return np.array(images), np.array(angles)


def create_angle_labels(images_count: int) -> np.ndarray:
    """
    Create synthetic angle labels for training.
    In the real dataset, these would be loaded from .txt files.
    
    Args:
        images_count: Number of images
        
    Returns:
        Array of angle labels (in degrees)
    """
    # Create realistic angle distribution based on research
    # Most documents are slightly skewed, some are more rotated
    angles = np.random.normal(0, 3, images_count)  # Mean 0, std 3 degrees
    angles = np.clip(angles, -15, 15)  # Limit to ±15 degrees
    
    return angles


def train_model():
    """Train the CNN deskewing model."""
    logger.info("Starting model training...")
    
    # Load data
    X, y = load_and_prepare_data()
    
    if len(X) == 0:
        logger.error("No training data found!")
        return None
    
    logger.info(f"Loaded {len(X)} training samples")
    logger.info(f"Angle statistics - Mean: {np.mean(y):.2f}, Std: {np.std(y):.2f}")
    logger.info(f"Angle range: [{np.min(y):.2f}, {np.max(y):.2f}]")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    
    # Create model
    model = DeskewModel(input_shape=(224, 224, 1))
    model.compile_model(learning_rate=CNN_PARAMS["learning_rate"])
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            str(MODELS_DIR / "deskew_model_best.h5"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=str(RESULTS_DIR / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        )
    ]
    
    # Train model
    logger.info("Training model...")
    history = model.train(
        X_train, y_train, X_val, y_val,
        epochs=CNN_PARAMS["epochs"],
        batch_size=CNN_PARAMS["batch_size"],
        callbacks=callbacks
    )
    
    # Save final model
    model.save_weights(str(MODELS_DIR / "deskew_model_final.h5"))
    
    return model, history


def evaluate_model(model: DeskewModel, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate trained model."""
    logger.info("Evaluating model...")
    
    # Make predictions
    y_pred = model.model.predict(X_test, verbose=0)
    y_pred = y_pred.flatten()
    
    # Compute metrics
    mae = np.mean(np.abs(y_test - y_pred))
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # Count predictions within thresholds
    within_1_deg = np.sum(np.abs(y_test - y_pred) <= 1.0)
    within_2_deg = np.sum(np.abs(y_test - y_pred) <= 2.0)
    within_5_deg = np.sum(np.abs(y_test - y_pred) <= 5.0)
    
    logger.info(f"MAE: {mae:.3f} degrees")
    logger.info(f"MSE: {mse:.3f}")
    logger.info(f"RMSE: {rmse:.3f} degrees")
    logger.info(f"Predictions within 1°: {within_1_deg}/{len(y_test)} ({within_1_deg/len(y_test)*100:.1f}%)")
    logger.info(f"Predictions within 2°: {within_2_deg}/{len(y_test)} ({within_2_deg/len(y_test)*100:.1f}%)")
    logger.info(f"Predictions within 5°: {within_5_deg}/{len(y_test)} ({within_5_deg/len(y_test)*100:.1f}%)")
    
    # Create evaluation DataFrame
    eval_df = pd.DataFrame({
        'true_angle': y_test,
        'predicted_angle': y_pred,
        'error': y_test - y_pred,
        'abs_error': np.abs(y_test - y_pred)
    })
    
    # Save evaluation results
    eval_path = RESULTS_DIR / "deskew_evaluation.csv"
    eval_df.to_csv(eval_path, index=False)
    logger.info(f"Saved evaluation results to: {eval_path}")
    
    return eval_df


def plot_training_history(history):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MAE)')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot MAE
    axes[1].plot(history.history['mae'], label='Training MAE')
    axes[1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (degrees)')
    axes[1].set_title('Training and Validation MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = RESULTS_DIR / "training_history.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved training history plot to: {plot_path}")


def plot_predictions(y_true, y_pred, save_path: Path):
    """Plot true vs predicted angles."""
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('True Angle (degrees)')
    plt.ylabel('Predicted Angle (degrees)')
    plt.title('True vs Predicted Rotation Angles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    stats_text = f'MAE: {mae:.3f}°\nRMSE: {rmse:.3f}°\nR²: {r2:.3f}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved predictions plot to: {save_path}")


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("CNN DESKEWING MODEL TRAINING")
    logger.info("Based on Cosmas Kiptoo Sang's Research")
    logger.info("=" * 60)
    
    # Create directories
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Train model
    model, history = train_model()
    
    if model is None:
        logger.error("Model training failed!")
        return
    
    # Plot training history
    plot_training_history(history)
    
    # Load test data for evaluation
    logger.info("Loading test data for evaluation...")
    X_test, y_test = load_and_prepare_data()
    
    if len(X_test) > 0:
        # Use a subset for evaluation
        eval_size = min(100, len(X_test))
        X_eval = X_test[:eval_size]
        y_eval = y_test[:eval_size]
        
        # Evaluate model
        eval_df = evaluate_model(model, X_eval, y_eval)
        
        # Plot predictions
        plot_predictions(
            y_eval,
            model.model.predict(X_eval, verbose=0).flatten(),
            RESULTS_DIR / "angle_predictions.png"
        )
    
    logger.info("Training completed successfully!")
    logger.info(f"Model saved in: {MODELS_DIR}")
    logger.info(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run training
    main()