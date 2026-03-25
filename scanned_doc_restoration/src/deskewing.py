"""
Deskewing Module for Scanned Document Restoration Pipeline
CNN-based rotation angle prediction for document deskewing
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, 
    GlobalAveragePooling2D, Dense, Dropout, Flatten, Input
)
import cv2
import numpy as np
from pathlib import Path
import json
import logging
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class DeskewModel:
    """CNN model for document deskewing angle prediction."""
    
    def __init__(self, input_shape: tuple = (224, 224, 1)):
        """
        Initialize deskewing model.
        
        Args:
            input_shape: Input shape (height, width, channels)
        """
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def build_model(self) -> tf.keras.Model:
        """
        Build CNN model for angle regression.
        
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # Input layer
            Input(shape=self.input_shape),
            
            # First conv block
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Second conv block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Third conv block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Fourth conv block
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Global pooling and dense layers
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='linear')  # Regression output for angle prediction
        ])
        
        return model
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile the model with Adam optimizer and MAE loss."""
        self.model = self.build_model()
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mean_absolute_error',
            metrics=['mae', 'mse']
        )
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32, callbacks=None):
        """
        Train the deskewing model.
        
        Args:
            X_train: Training images
            y_train: Training angles
            X_val: Validation images
            y_val: Validation angles
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: Keras callbacks
        
        Returns:
            Training history
        """
        if self.model is None:
            self.compile_model()
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history.history
        return history
    
    def predict_angle(self, image: np.ndarray) -> float:
        """
        Predict rotation angle for a single image.
        
        Args:
            image: Input image (grayscale, 224x224)
            
        Returns:
            Predicted angle in degrees
        """
        if self.model is None:
            raise ValueError("Model not compiled. Call compile_model() first.")
        
        # Preprocess image
        img = self.preprocess_image(image)
        
        # Add batch dimension
        img_batch = np.expand_dims(img, axis=0)
        
        # Predict
        prediction = self.model.predict(img_batch, verbose=0)
        return float(prediction[0][0])
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for the model.
        
        Args:
            image: Input image (grayscale, any size)
            
        Returns:
            Preprocessed image (224x224, normalized)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image = image[:, :, 0]  # Take first channel if single channel
        
        # Resize to model input size
        image = cv2.resize(image, (224, 224))
        
        # Normalize to [0, 1]
        image = image.astype('float32') / 255.0
        
        # Add channel dimension
        image = np.expand_dims(image, axis=-1)
        
        return image
    
    def load_weights(self, weights_path: str):
        """Load pre-trained weights."""
        if self.model is None:
            self.compile_model()
        
        # Check if path exists, if not try with .weights.h5 extension
        weights_path = Path(weights_path)
        if not weights_path.exists():
            # Try with .weights.h5 extension
            alt_path = weights_path.parent / f"{weights_path.stem}.weights.h5"
            if alt_path.exists():
                weights_path = alt_path
        
        self.model.load_weights(str(weights_path))
        print(f"Loaded weights from {weights_path}")
    
    def save_weights(self, weights_path: str):
        """Save model weights."""
        if self.model is not None:
            # Ensure .weights.h5 extension
            weights_path = Path(weights_path)
            if not str(weights_path).endswith('.weights.h5'):
                weights_path = weights_path.parent / f"{weights_path.stem}.weights.h5"
            
            self.model.save_weights(str(weights_path))
            print(f"Saved weights to {weights_path}")
        else:
            raise ValueError("Model not compiled. Call compile_model() first.")


class DeskewPipeline:
    """Complete deskewing pipeline."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize deskewing pipeline.
        
        Args:
            model_path: Path to pre-trained model weights
        """
        self.model = DeskewModel()
        self.model.compile_model()
        
        if model_path and Path(model_path).exists():
            self.model.load_weights(model_path)
    
    def deskew_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Deskew an image by predicting and correcting its rotation.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Tuple of (deskewed_image, predicted_angle)
        """
        # Predict angle
        angle = self.model.predict_angle(image)
        
        # Rotate image to correct angle
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
        deskewed = cv2.warpAffine(image, rotation_matrix, 
                                  (image.shape[1], image.shape[0]))
        
        return deskewed, angle
    
    def batch_deskew(self, images: list) -> Tuple[list, list]:
        """
        Deskew a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            Tuple of (deskewed_images, predicted_angles)
        """
        deskewed_images = []
        angles = []
        
        for img in images:
            deskewed_img, angle = self.deskew_image(img)
            deskewed_images.append(deskewed_img)
            angles.append(angle)
        
        return deskewed_images, angles


def load_training_data(data_dir: Path, labels_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data from directory.
    
    Args:
        data_dir: Directory containing training images
        labels_file: JSON file with angle labels
        
    Returns:
        Tuple of (images, angles)
    """
    images = []
    angles = []
    
    # Load labels
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    
    for img_name, angle in labels.items():
        img_path = data_dir / f"{img_name}.png"
        if img_path.exists():
            # Load and preprocess image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
            
            images.append(img)
            angles.append(angle)
    
    return np.array(images), np.array(angles)


def train_deskew_model(data_dir: Path, labels_file: Path, 
                       model_save_path: Path, epochs: int = 50):
    """
    Train the deskewing model.
    
    Args:
        data_dir: Directory with training images
        labels_file: JSON file with angle labels
        model_save_path: Where to save the trained model
        epochs: Number of training epochs
    """
    # Load data
    X, y = load_training_data(data_dir, labels_file)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and compile model
    deskew_model = DeskewModel()
    model = deskew_model.compile_model()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(model_save_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        )
    ]
    
    # Train model
    history = deskew_model.train(
        X_train, y_train, X_val, y_val,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return deskew_model, history


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from config import TRAIN_IMAGES_DIR, TRAIN_LABELS_PATH
    
    # Initialize model
    deskewer = DeskewPipeline()
    
    # Example: Deskew a single image
    test_image = cv2.imread("test_image.png", cv2.IMREAD_GRAYSCALE)
    deskewed, angle = deskewer.deskew_image(test_image)
    
    print(f"Predicted angle: {angle:.2f} degrees")
    print(f"Deskewed image shape: {deskewed.shape}")