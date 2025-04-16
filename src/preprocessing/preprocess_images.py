#!/usr/bin/env python
"""
Preprocess MODIS satellite imagery for flood detection.
This module handles:
1. Reading GeoTIFF files
2. Image enhancement and normalization
3. Cloud and shadow masking
4. Creating training/validation datasets
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import geopandas as gpd
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm

# Set constants for image processing
VALID_DATA_RANGE = (0, 255)  # Valid range for MODIS flood products
WATER_CLASS = 1  # Value representing water in MODIS flood products
NO_WATER_CLASS = 0  # Value representing non-water areas
CLOUD_MASKED = 2  # Value representing cloud-masked pixels
NO_DATA = 255  # No data value

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess MODIS flood mapping data')
    parser.add_argument('--input_dir', type=str, 
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                          'data', 'modis', '2D2OT'),
                        help='Directory containing downloaded MODIS flood data')
    parser.add_argument('--output_dir', type=str, 
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                           'data', 'processed'),
                        help='Output directory for processed images')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Size of image patches for training')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    return parser.parse_args()

def list_geotiff_files(input_dir):
    """List all GeoTIFF files in the input directory."""
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    
    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))
    print(f"Found {len(tif_files)} GeoTIFF files in {input_dir}")
    return tif_files

def read_geotiff(file_path):
    """Read a GeoTIFF file and return the image array and metadata."""
    try:
        with rasterio.open(file_path) as src:
            image = src.read(1)  # Read the first band
            metadata = src.meta.copy()
            return image, metadata
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None, None

def create_water_mask(image):
    """Create a binary water mask from MODIS flood classification."""
    # Create a copy to avoid modifying the original
    water_mask = np.copy(image)
    
    # Set water pixels to 1, all others to 0
    water_mask[water_mask == WATER_CLASS] = 1
    water_mask[water_mask != 1] = 0
    
    return water_mask.astype(np.uint8)

def create_valid_data_mask(image):
    """Create a mask of valid data (not cloud or no-data)."""
    valid_mask = np.ones_like(image, dtype=np.uint8)
    valid_mask[image == CLOUD_MASKED] = 0
    valid_mask[image == NO_DATA] = 0
    
    return valid_mask

def enhance_image(image, valid_mask):
    """Apply image enhancement techniques."""
    # Normalize to 0-255 range
    normalized = np.zeros_like(image, dtype=np.uint8)
    valid_pixels = image[valid_mask == 1]
    
    if len(valid_pixels) > 0:
        min_val = np.min(valid_pixels)
        max_val = np.max(valid_pixels)
        
        if max_val > min_val:
            # Linear contrast stretch
            normalized = np.clip(((image - min_val) / (max_val - min_val) * 255), 0, 255).astype(np.uint8)
            
            # Apply histogram equalization to valid areas
            normalized[valid_mask == 1] = cv2.equalizeHist(normalized[valid_mask == 1])
    
    return normalized

def generate_patches(image, mask, patch_size, overlap=0.5):
    """Generate overlapping patches from image and mask."""
    height, width = image.shape
    stride = int(patch_size * (1 - overlap))
    
    patches = []
    mask_patches = []
    locations = []
    
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            img_patch = image[y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]
            
            # Only keep patches with sufficient valid data
            valid_ratio = np.sum(mask_patch > 0) / (patch_size * patch_size)
            if valid_ratio >= 0.7:  # At least 70% valid data
                patches.append(img_patch)
                mask_patches.append(mask_patch)
                locations.append((x, y))
    
    return patches, mask_patches, locations

def save_patch_dataset(patches, masks, output_dir, prefix="train"):
    """Save image patches and masks as numpy arrays."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    patches_array = np.array(patches)
    masks_array = np.array(masks)
    
    # Save as numpy arrays
    np.save(os.path.join(output_dir, f"{prefix}_patches.npy"), patches_array)
    np.save(os.path.join(output_dir, f"{prefix}_masks.npy"), masks_array)
    
    print(f"Saved {len(patches)} {prefix} patches to {output_dir}")
    
    # Save sample images for visual inspection
    sample_dir = os.path.join(output_dir, f"{prefix}_samples")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    
    # Save a few samples for visual inspection
    for i in range(min(10, len(patches))):
        cv2.imwrite(os.path.join(sample_dir, f"sample_{i}_image.png"), patches[i])
        cv2.imwrite(os.path.join(sample_dir, f"sample_{i}_mask.png"), masks[i] * 255)  # Scale mask for visibility

def process_file(file_path, output_dir, patch_size):
    """Process a single GeoTIFF file."""
    filename = os.path.basename(file_path)
    print(f"Processing {filename}...")
    
    # Read the file
    image, metadata = read_geotiff(file_path)
    if image is None:
        return None, None
    
    # Create water mask
    water_mask = create_water_mask(image)
    
    # Create valid data mask
    valid_mask = create_valid_data_mask(image)
    
    # Enhance image for visualization and feature extraction
    enhanced_image = enhance_image(image, valid_mask)
    
    # Generate patches for machine learning
    patches, mask_patches, locations = generate_patches(enhanced_image, water_mask, patch_size)
    
    # Save intermediate results
    output_filename = os.path.splitext(filename)[0]
    intermediate_dir = os.path.join(output_dir, "intermediate", output_filename)
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)
    
    # Save the processed masks and enhanced images
    with rasterio.open(os.path.join(intermediate_dir, "water_mask.tif"), 'w', **metadata) as dst:
        dst.write(water_mask, 1)
    
    with rasterio.open(os.path.join(intermediate_dir, "valid_mask.tif"), 'w', **metadata) as dst:
        dst.write(valid_mask, 1)
    
    with rasterio.open(os.path.join(intermediate_dir, "enhanced.tif"), 'w', **metadata) as dst:
        dst.write(enhanced_image, 1)
    
    return patches, mask_patches

def main():
    """Main function for preprocessing MODIS flood data."""
    args = parse_arguments()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # List all GeoTIFF files
    tif_files = list_geotiff_files(args.input_dir)
    
    # Process each file
    all_patches = []
    all_masks = []
    
    for file_path in tqdm(tif_files, desc="Processing files"):
        patches, masks = process_file(file_path, args.output_dir, args.patch_size)
        if patches is not None:
            all_patches.extend(patches)
            all_masks.extend(masks)
    
    print(f"Total patches generated: {len(all_patches)}")
    
    # Split into training and test sets
    if len(all_patches) > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            all_patches, all_masks, test_size=args.test_split, random_state=42
        )
        
        # Save the datasets
        save_patch_dataset(X_train, y_train, args.output_dir, prefix="train")
        save_patch_dataset(X_test, y_test, args.output_dir, prefix="test")
        
        print(f"Preprocessing complete. Training set: {len(X_train)}, Test set: {len(X_test)}")
    else:
        print("No valid patches were generated. Check the input data.")

if __name__ == "__main__":
    main() 