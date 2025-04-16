#!/usr/bin/env python
"""
Flood Prediction Demo

This script demonstrates the flood prediction system with sample data.
It simulates the process without requiring real satellite imagery.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
import folium
from folium.plugins import HeatMap
from shapely.geometry import Polygon
import geopandas as gpd

# Create sample directories
def create_dirs():
    """Create directories for the demo."""
    dirs = ['data/sample', 'models/demo', 'outputs/demo']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs

# Generate synthetic satellite imagery
def generate_sample_data(output_dir, num_samples=5, size=512):
    """Generate synthetic flood data for demonstration."""
    print("Generating synthetic flood data...")
    
    # Parameters for simulating different flood patterns
    patterns = [
        {'radius': size//4, 'num_centers': 3, 'noise': 0.1},  # Multiple water bodies
        {'radius': size//3, 'num_centers': 1, 'noise': 0.05},  # Single large water body
        {'radius': size//6, 'num_centers': 6, 'noise': 0.15},  # Many small water bodies
        {'radius': size//2, 'num_centers': 1, 'noise': 0.2},   # Large water body with noise
        {'radius': size//5, 'num_centers': 4, 'noise': 0.1}    # Medium water bodies
    ]
    
    filepaths = []
    
    # Create a transform for GeoTIFF (arbitrary coordinates for demo)
    transform = from_origin(0, 0, 0.01, 0.01)
    
    for i in tqdm(range(num_samples)):
        # Select a pattern
        pattern = patterns[i % len(patterns)]
        
        # Create an empty image
        image = np.zeros((size, size), dtype=np.uint8)
        
        # Create a ground truth water mask
        water_mask = np.zeros((size, size), dtype=np.uint8)
        
        # Add water bodies
        for _ in range(pattern['num_centers']):
            # Random center
            center_x = np.random.randint(0, size)
            center_y = np.random.randint(0, size)
            
            # Random radius with variation
            radius = int(pattern['radius'] * (0.7 + 0.6 * np.random.random()))
            
            # Draw water body
            cv2.circle(water_mask, (center_x, center_y), radius, 1, -1)
        
        # Add some random noise to make it more natural-looking
        noise = np.random.random((size, size)) * pattern['noise']
        water_mask_float = water_mask.astype(float)
        water_mask_float += noise
        water_mask = (water_mask_float > 0.5).astype(np.uint8)
        
        # Create image with classes similar to MODIS flood data
        # 0: no water, 1: water, 2: cloud-masked, 255: no data
        image[water_mask == 1] = 1  # Water
        
        # Add some cloud cover (random patches)
        num_clouds = np.random.randint(1, 4)
        for _ in range(num_clouds):
            cloud_x = np.random.randint(0, size)
            cloud_y = np.random.randint(0, size)
            cloud_radius = np.random.randint(size//10, size//5)
            cv2.circle(image, (cloud_x, cloud_y), cloud_radius, 2, -1)
        
        # Add some no-data areas (random patches)
        num_nodata = np.random.randint(0, 2)
        for _ in range(num_nodata):
            nodata_x = np.random.randint(0, size)
            nodata_y = np.random.randint(0, size)
            nodata_radius = np.random.randint(size//10, size//5)
            cv2.circle(image, (nodata_x, nodata_y), nodata_radius, 255, -1)
        
        # Save as GeoTIFF
        output_path = os.path.join(output_dir, f'sample_flood_{i+1}.tif')
        
        # Create metadata for the GeoTIFF
        metadata = {
            'driver': 'GTiff',
            'height': size,
            'width': size,
            'count': 1,
            'dtype': rasterio.uint8,
            'crs': '+proj=latlong',
            'transform': transform
        }
        
        with rasterio.open(output_path, 'w', **metadata) as dst:
            dst.write(image, 1)
        
        filepaths.append(output_path)
        
        # Save ground truth for testing
        truth_path = os.path.join(output_dir, f'sample_flood_{i+1}_truth.tif')
        with rasterio.open(truth_path, 'w', **metadata) as dst:
            dst.write(water_mask, 1)
    
    print(f"Generated {num_samples} sample images in {output_dir}")
    return filepaths

# Create a small U-Net model for demo
def create_demo_unet(input_shape=(256, 256, 1)):
    """Create a small U-Net model for demonstration."""
    # Input layer
    inputs = Input(input_shape)
    
    # Encoder (simplified)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Bridge
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    drop3 = Dropout(0.5)(conv3)
    
    # Decoder (simplified)
    up4 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop3))
    merge4 = concatenate([conv2, up4], axis=3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(merge4)
    
    up5 = Conv2D(32, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv4))
    merge5 = concatenate([conv1, up5], axis=3)
    conv5 = Conv2D(32, 3, activation='relu', padding='same')(merge5)
    
    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv5)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Process an image and predict flood areas
def process_and_predict(image_path, model, output_dir, patch_size=256):
    """Process a sample image and generate a flood prediction."""
    # Read the image
    with rasterio.open(image_path) as src:
        image = src.read(1)
        metadata = src.meta.copy()
    
    # Create enhanced image (simple contrast enhancement for demo)
    enhanced = np.copy(image).astype(np.float32)
    mask = (enhanced != 2) & (enhanced != 255)  # Not cloud or no-data
    if np.any(mask):
        min_val = np.min(enhanced[mask])
        max_val = np.max(enhanced[mask])
        if max_val > min_val:
            enhanced[mask] = (enhanced[mask] - min_val) / (max_val - min_val)
    enhanced = (enhanced * 255).astype(np.uint8)
    
    # Extract patches
    height, width = image.shape
    patches = []
    positions = []
    
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            # Handle boundary cases
            y_end = min(y + patch_size, height)
            x_end = min(x + patch_size, width)
            
            patch = enhanced[y:y_end, x:x_end]
            
            # Pad if necessary
            if patch.shape != (patch_size, patch_size):
                padded = np.zeros((patch_size, patch_size), dtype=patch.dtype)
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded
            
            patches.append(patch)
            positions.append((y, x))
    
    # Normalize and reshape for prediction
    patches_array = np.array(patches) / 255.0
    patches_array = np.expand_dims(patches_array, axis=-1)
    
    # Predict
    predictions = model.predict(patches_array)
    
    # Stitch predictions
    prediction_map = np.zeros(image.shape, dtype=np.float32)
    for pred, (y, x) in zip(predictions, positions):
        pred = pred.squeeze()
        
        # Handle boundary cases
        y_end = min(y + patch_size, height)
        x_end = min(x + patch_size, width)
        pred_height = y_end - y
        pred_width = x_end - x
        
        prediction_map[y:y_end, x:x_end] = pred[:pred_height, :pred_width]
    
    # Create binary flood map
    flood_map = (prediction_map >= 0.5).astype(np.uint8)
    
    # Save results
    output_filename = os.path.basename(image_path).replace('.tif', '')
    
    # Save prediction map
    pred_path = os.path.join(output_dir, f"{output_filename}_prediction.tif")
    with rasterio.open(pred_path, 'w', **metadata) as dst:
        dst.write(flood_map, 1)
    
    # Create visualization
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='viridis')
    plt.title('Original Image')
    plt.axis('off')
    
    # Enhanced image
    plt.subplot(1, 3, 2)
    plt.imshow(enhanced, cmap='gray')
    plt.title('Enhanced Image')
    plt.axis('off')
    
    # Prediction
    plt.subplot(1, 3, 3)
    plt.imshow(flood_map, cmap='Blues')
    plt.title('Flood Prediction')
    plt.axis('off')
    
    plt.tight_layout()
    vis_path = os.path.join(output_dir, f"{output_filename}_visualization.png")
    plt.savefig(vis_path)
    plt.close()
    
    return pred_path, vis_path

# Create an interactive map
def create_interactive_map(prediction_path, output_dir):
    """Create an interactive map from the prediction."""
    # Read the prediction
    with rasterio.open(prediction_path) as src:
        prediction = src.read(1)
        transform = src.transform
        crs = src.crs
    
    # Sample coordinates for demo - typically these would come from the GeoTIFF
    # Since our demo images have arbitrary coordinates, we'll use some example ones
    lat, lon = 30.0, -90.0  # Example: Gulf of Mexico
    
    # Create a map
    m = folium.Map(location=[lat, lon], zoom_start=8)
    
    # For demonstration, create some polygons based on the prediction
    # In a real scenario, we'd vectorize the prediction raster
    height, width = prediction.shape
    polygons = []
    
    # Simplified approach: just create a few random polygons where prediction is positive
    sample_points = []
    for _ in range(5):
        # Find random areas with water
        y, x = np.where(prediction == 1)
        if len(y) > 0:
            idx = np.random.randint(0, len(y))
            center_y, center_x = y[idx], x[idx]
            
            # Create a polygon around this point
            size = np.random.randint(3, 10)
            points = []
            for i in range(6):  # Hexagon
                angle = i * 2 * np.pi / 6
                offset_y = size * np.sin(angle)
                offset_x = size * np.cos(angle)
                points.append((lat + offset_y * 0.01, lon + offset_x * 0.01))
            
            polygons.append(Polygon(points))
            
            # Add some sample points for the heatmap
            for _ in range(20):
                sample_lat = lat + (np.random.random() - 0.5) * 0.1
                sample_lon = lon + (np.random.random() - 0.5) * 0.1
                intensity = np.random.random()
                sample_points.append([sample_lat, sample_lon, intensity])
    
    # Create GeoDataFrame
    if polygons:
        gdf = gpd.GeoDataFrame({'geometry': polygons})
        gdf['area'] = gdf.geometry.area
        
        # Add to map
        folium.GeoJson(
            gdf,
            name='Flood Areas',
            style_function=lambda x: {
                'fillColor': 'blue',
                'color': 'blue',
                'weight': 1,
                'fillOpacity': 0.5
            }
        ).add_to(m)
    
    # Add heatmap
    if sample_points:
        HeatMap(sample_points).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map
    map_path = os.path.join(output_dir, "interactive_flood_map.html")
    m.save(map_path)
    
    return map_path

# Main function to run the demo
def main():
    """Run the flood prediction demo."""
    print("Starting Flood Prediction Demo")
    
    # Create directories
    dirs = create_dirs()
    data_dir, model_dir, output_dir = dirs
    
    # Generate sample data
    sample_dir = os.path.join(data_dir, 'sample')
    filepaths = generate_sample_data(sample_dir)
    
    # Create and train a small model
    print("\nCreating and training a small U-Net model...")
    model = create_demo_unet()
    model.summary()
    
    # For demo, we'll train a very simple model on our synthetic data
    X_train = []
    y_train = []
    
    # Use a subset of the generated data for training
    for i in range(min(3, len(filepaths))):
        # Read image
        with rasterio.open(filepaths[i]) as src:
            image = src.read(1)
        
        # Read ground truth
        truth_path = filepaths[i].replace('.tif', '_truth.tif')
        with rasterio.open(truth_path) as src:
            truth = src.read(1)
        
        # Simple preprocessing: normalize and create patches
        image_norm = image.astype(np.float32) / 255.0
        
        # Create a few simple patches
        for _ in range(10):
            h, w = image.shape
            y = np.random.randint(0, h - 256)
            x = np.random.randint(0, w - 256)
            
            patch = image_norm[y:y+256, x:x+256]
            mask = truth[y:y+256, x:x+256]
            
            X_train.append(patch)
            y_train.append(mask)
    
    X_train = np.array(X_train).reshape(-1, 256, 256, 1)
    y_train = np.array(y_train).reshape(-1, 256, 256, 1)
    
    # Train for just a few epochs (demo only)
    model.fit(X_train, y_train, batch_size=4, epochs=5, verbose=1)
    
    # Save the model
    model_path = os.path.join(model_dir, 'demo_unet_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Process and predict
    print("\nGenerating flood predictions...")
    prediction_paths = []
    
    # Use the remaining images for prediction
    for i in range(min(3, len(filepaths)), len(filepaths)):
        pred_path, vis_path = process_and_predict(filepaths[i], model, output_dir)
        prediction_paths.append(pred_path)
        print(f"Processed {filepaths[i]}, visualization saved to {vis_path}")
    
    # Create an interactive map
    if prediction_paths:
        map_path = create_interactive_map(prediction_paths[0], output_dir)
        print(f"\nInteractive map saved to {map_path}")
    
    print("\nDemo completed successfully!")
    print(f"Output files are in {output_dir}")

if __name__ == "__main__":
    main() 