#!/usr/bin/env python
"""
Train a U-Net model for flood prediction and mapping from satellite imagery.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train flood prediction model')
    parser.add_argument('--data_dir', type=str, 
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                           'data', 'processed'),
                        help='Directory containing processed image data')
    parser.add_argument('--output_dir', type=str, 
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                            'models'),
                        help='Output directory for trained models')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    return parser.parse_args()

def load_data(data_dir):
    """Load training and test data."""
    train_patches_path = os.path.join(data_dir, "train_patches.npy")
    train_masks_path = os.path.join(data_dir, "train_masks.npy")
    test_patches_path = os.path.join(data_dir, "test_patches.npy")
    test_masks_path = os.path.join(data_dir, "test_masks.npy")
    
    if not os.path.exists(train_patches_path) or not os.path.exists(train_masks_path):
        raise FileNotFoundError(f"Training data not found in {data_dir}")
    
    # Load training data
    X_train = np.load(train_patches_path)
    y_train = np.load(train_masks_path)
    
    # Load test data if available
    if os.path.exists(test_patches_path) and os.path.exists(test_masks_path):
        X_test = np.load(test_patches_path)
        y_test = np.load(test_masks_path)
    else:
        print("Test data not found, using a portion of training data for testing")
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
    
    # Reshape data for model input if needed
    if len(X_train.shape) == 3:
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)
    
    if len(y_train.shape) == 3:
        y_train = np.expand_dims(y_train, -1)
        y_test = np.expand_dims(y_test, -1)
    
    # Normalize input data to 0-1 range
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, {y_test.shape}")
    
    return X_train, y_train, X_test, y_test

def create_unet_model(input_shape, num_classes=1):
    """Create a U-Net model for semantic segmentation."""
    # Input layer
    inputs = Input(input_shape)
    
    # Encoder (Contracting Path)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    # Bridge
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    # Decoder (Expanding Path)
    up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    # Output layer
    outputs = Conv2D(num_classes, 1, activation='sigmoid')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def dice_coefficient(y_true, y_pred, smooth=1.0):
    """Calculate Dice coefficient for model evaluation."""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss function for training."""
    return 1.0 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """Combined binary crossentropy and dice loss."""
    bce = tf.keras.losses.BinaryCrossentropy()
    return 0.5 * bce(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)

def plot_training_history(history, output_dir):
    """Plot and save training history."""
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot dice coefficient
    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coefficient'])
    plt.plot(history.history['val_dice_coefficient'])
    plt.title('Dice Coefficient')
    plt.ylabel('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def save_sample_predictions(model, X_test, y_test, output_dir, num_samples=5):
    """Save sample predictions for visual inspection."""
    samples_dir = os.path.join(output_dir, "prediction_samples")
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    
    # Get predictions
    y_pred = model.predict(X_test[:num_samples])
    
    for i in range(num_samples):
        plt.figure(figsize=(12, 4))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(X_test[i, :, :, 0], cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Ground truth
        plt.subplot(1, 3, 2)
        plt.imshow(y_test[i, :, :, 0], cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        
        # Prediction
        plt.subplot(1, 3, 3)
        plt.imshow(y_pred[i, :, :, 0], cmap='gray')
        plt.title('Prediction')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(samples_dir, f'sample_prediction_{i}.png'))
        plt.close()

def main():
    """Main function for training flood prediction model."""
    args = parse_arguments()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load data
    X_train, y_train, X_test, y_test = load_data(args.data_dir)
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=args.val_split, random_state=42
    )
    
    # Create model
    input_shape = X_train.shape[1:]
    model = create_unet_model(input_shape)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss=combined_loss,
        metrics=[dice_coefficient]
    )
    
    model.summary()
    
    # Set up callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.output_dir, f"flood_unet_model_{timestamp}.h5")
    
    callbacks = [
        ModelCheckpoint(model_path, monitor='val_dice_coefficient', 
                         mode='max', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, 
                          min_lr=1e-6, verbose=1),
        TensorBoard(log_dir=os.path.join(args.output_dir, 'logs', timestamp))
    ]
    
    # Train model
    print(f"Training model with {X_train.shape[0]} samples...")
    history = model.fit(
        X_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    # Evaluate model
    test_loss, test_dice = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test Dice Coefficient: {test_dice:.4f}")
    
    # Save model and metrics
    model.save(os.path.join(args.output_dir, "flood_unet_final_model.h5"))
    
    # Plot training history
    plot_training_history(history, args.output_dir)
    
    # Save sample predictions
    save_sample_predictions(model, X_test, y_test, args.output_dir)
    
    # Save model summary and parameters
    with open(os.path.join(args.output_dir, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write('\n\nTraining Parameters:\n')
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Final Test Loss: {test_loss:.4f}\n")
        f.write(f"Final Test Dice Coefficient: {test_dice:.4f}\n")
    
    print(f"Training complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main() 