#!/usr/bin/env python
"""
Flood Prediction and Mapping Using Satellite Imagery

This script runs the full pipeline for flood prediction and mapping:
1. Download MODIS flood data
2. Preprocess the images
3. Train a U-Net model
4. Generate flood prediction maps
"""

import os
import argparse
import subprocess
import datetime

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run flood prediction pipeline')
    parser.add_argument('--start_date', type=str, 
                        default=(datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d'),
                        help='Start date for data download (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, 
                        default=datetime.datetime.now().strftime('%Y-%m-%d'),
                        help='End date for data download (YYYY-MM-DD)')
    parser.add_argument('--region', type=str, default='global',
                        help='Region of interest (e.g., global, asia, africa)')
    parser.add_argument('--product_type', type=str, default='2D2OT',
                        choices=['1D1OS', '2D2OT', '3D3OT', '14DXT'],
                        help='MODIS flood product type')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip data download step')
    parser.add_argument('--skip_preprocessing', action='store_true',
                        help='Skip preprocessing step')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip model training step')
    parser.add_argument('--skip_prediction', action='store_true',
                        help='Skip prediction step')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training and prediction')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for training')
    return parser.parse_args()

def set_gpu_config(use_gpu):
    """Configure GPU/CPU usage for TensorFlow."""
    # Import TensorFlow here to avoid dependency if only doing data download
    import tensorflow as tf
    
    if use_gpu:
        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU is enabled. Found {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"Error setting up GPU: {e}")
        else:
            print("Warning: GPU requested but no GPU found. Using CPU instead.")
    else:
        # Disable GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("GPU is disabled. Using CPU.")

def run_command(command, description):
    """Run a shell command and print its output."""
    print(f"\n{'='*80}\n{description}\n{'='*80}")
    print(f"Running: {' '.join(command)}")
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Command completed successfully")
        print(result.stdout)
    else:
        print("Command failed")
        print(f"Error: {result.stderr}")
    
    return result.returncode == 0

def download_data(args):
    """Download MODIS flood data."""
    command = [
        "python", 
        "src/data_acquisition/download_modis_data.py",
        "--start_date", args.start_date,
        "--end_date", args.end_date,
        "--region", args.region,
        "--product_type", args.product_type
    ]
    
    return run_command(command, "Downloading MODIS flood data")

def preprocess_data(args):
    """Preprocess downloaded images."""
    input_dir = os.path.join("data", "modis", args.product_type)
    output_dir = os.path.join("data", "processed")
    
    command = [
        "python",
        "src/preprocessing/preprocess_images.py",
        "--input_dir", input_dir,
        "--output_dir", output_dir
    ]
    
    return run_command(command, "Preprocessing satellite images")

def train_model(args):
    """Train the U-Net model."""
    data_dir = os.path.join("data", "processed")
    output_dir = "models"
    
    command = [
        "python",
        "src/models/train_model.py",
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs)
    ]
    
    return run_command(command, "Training U-Net model")

def generate_predictions(args):
    """Generate flood prediction maps."""
    model_path = os.path.join("models", "flood_unet_final_model.h5")
    data_dir = os.path.join("data", "modis", args.product_type)
    output_dir = "outputs"
    
    command = [
        "python",
        "src/visualization/generate_flood_maps.py",
        "--model_path", model_path,
        "--data_dir", data_dir,
        "--output_dir", output_dir
    ]
    
    return run_command(command, "Generating flood prediction maps")

def main():
    """Main function to run the flood prediction pipeline."""
    args = parse_arguments()
    
    # Configure GPU if needed
    if not args.skip_training or not args.skip_prediction:
        set_gpu_config(args.gpu)
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Run the pipeline steps
    success = True
    
    if not args.skip_download:
        success = download_data(args)
        if not success:
            print("Data download failed, aborting pipeline")
            return
    
    if not args.skip_preprocessing and success:
        success = preprocess_data(args)
        if not success:
            print("Preprocessing failed, aborting pipeline")
            return
    
    if not args.skip_training and success:
        success = train_model(args)
        if not success:
            print("Model training failed, aborting pipeline")
            return
    
    if not args.skip_prediction and success:
        success = generate_predictions(args)
        if not success:
            print("Prediction generation failed")
            return
    
    if success:
        print("\nFlood prediction pipeline completed successfully!")
        print("Results are available in the 'outputs' directory")

if __name__ == "__main__":
    main() 