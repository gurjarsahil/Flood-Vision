#!/usr/bin/env python
"""
Generate flood prediction maps using the trained U-Net model.
This script:
1. Loads the trained model
2. Processes satellite imagery
3. Generates flood prediction maps
4. Creates visualizations and interactive maps
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tensorflow as tf
import rasterio
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, mapping
import folium
from folium.plugins import HeatMap
from tqdm import tqdm
import glob
from datetime import datetime
import pandas as pd

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate flood prediction maps')
    parser.add_argument('--model_path', type=str, 
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                           'models', 'flood_unet_final_model.h5'),
                        help='Path to trained model file')
    parser.add_argument('--data_dir', type=str, 
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                           'data', 'modis', '2D2OT'),
                        help='Directory containing satellite images to process')
    parser.add_argument('--output_dir', type=str, 
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                            'outputs'),
                        help='Output directory for generated maps')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Size of image patches for prediction')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap between patches (0-1)')
    parser.add_argument('--water_threshold', type=float, default=0.5,
                        help='Threshold for water classification (0-1)')
    return parser.parse_args()

def load_model(model_path):
    """Load the trained U-Net model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Custom objects needed for loading the model
    custom_objects = {
        'dice_coefficient': dice_coefficient,
        'dice_loss': dice_loss,
        'combined_loss': combined_loss
    }
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

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

def list_image_files(data_dir, extension='.tif'):
    """List all image files in the data directory."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    files = glob.glob(os.path.join(data_dir, f"*{extension}"))
    print(f"Found {len(files)} {extension} files in {data_dir}")
    return files

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

def preprocess_image(image):
    """Preprocess the image for model input."""
    # Create a copy to avoid modifying the original
    processed = np.copy(image)
    
    # Handle no-data values
    no_data_mask = (processed == 255)
    processed = processed.astype(np.float32)
    
    # Normalize to 0-1 range
    min_val = np.min(processed[~no_data_mask]) if np.any(~no_data_mask) else 0
    max_val = np.max(processed[~no_data_mask]) if np.any(~no_data_mask) else 1
    
    if max_val > min_val:
        processed = (processed - min_val) / (max_val - min_val)
    
    # Set no-data to 0
    processed[no_data_mask] = 0
    
    return processed

def generate_patches(image, patch_size, overlap=0.5):
    """Generate overlapping patches from image."""
    height, width = image.shape
    stride = int(patch_size * (1 - overlap))
    
    # Calculate number of patches in each dimension
    n_h = max(1, (height - patch_size) // stride + 1)
    n_w = max(1, (width - patch_size) // stride + 1)
    
    patches = []
    positions = []
    
    for i in range(n_h):
        for j in range(n_w):
            y = min(i * stride, height - patch_size)
            x = min(j * stride, width - patch_size)
            
            patch = image[y:y+patch_size, x:x+patch_size]
            
            # Pad if necessary
            if patch.shape != (patch_size, patch_size):
                padded = np.zeros((patch_size, patch_size), dtype=patch.dtype)
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded
            
            patches.append(patch)
            positions.append((y, x))
    
    return np.array(patches), positions

def stitch_predictions(predictions, positions, original_shape, patch_size, overlap=0.5):
    """Stitch together predictions from patches into a single image."""
    height, width = original_shape
    stride = int(patch_size * (1 - overlap))
    
    # Initialize output array and weight map
    output = np.zeros(original_shape, dtype=np.float32)
    weight = np.zeros(original_shape, dtype=np.float32)
    
    # Create a weight mask for blending overlapping regions
    # Higher weights in the center, lower at the edges
    y, x = np.mgrid[0:patch_size, 0:patch_size]
    y = np.minimum(y, patch_size - y - 1)
    x = np.minimum(x, patch_size - x - 1)
    mask = np.minimum(y, x) + 1  # +1 to avoid zero weights
    
    # Apply predictions with weights
    for pred, (y, x) in zip(predictions, positions):
        pred = pred.squeeze()  # Remove channel dimension if present
        
        # Handle boundary cases
        h = min(patch_size, height - y)
        w = min(patch_size, width - x)
        
        # Add weighted prediction
        output[y:y+h, x:x+w] += pred[:h, :w] * mask[:h, :w]
        weight[y:y+h, x:x+w] += mask[:h, :w]
    
    # Normalize by weights
    valid_mask = weight > 0
    output[valid_mask] /= weight[valid_mask]
    
    return output

def create_flood_map(prediction, threshold=0.5):
    """Create a binary flood map from prediction probabilities."""
    # Threshold the prediction
    flood_map = (prediction >= threshold).astype(np.uint8)
    return flood_map

def save_geotiff(data, metadata, output_path):
    """Save data as a GeoTIFF file."""
    # Update metadata for the output
    metadata.update({
        'count': 1,
        'dtype': data.dtype,
        'compress': 'lzw'
    })
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the output
    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(data, 1)
    
    print(f"Saved GeoTIFF to {output_path}")
    return output_path

def create_colormap_visualization(image, flood_map, output_path):
    """Create a colored visualization of the flood map."""
    # Create a colormap for visualization
    # Blue for water, green for land, red for flooded areas
    cmap = mcolors.ListedColormap(['darkgreen', 'blue'])
    bounds = [0, 0.5, 1]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    plt.figure(figsize=(12, 8))
    
    # Plot as an RGB image
    plt.imshow(flood_map, cmap=cmap, norm=norm, interpolation='none')
    plt.colorbar(ticks=[0.25, 0.75], label='Class')
    plt.clim(0, 1)
    plt.title('Flood Prediction Map')
    plt.axis('off')
    
    # Save the image
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {output_path}")
    return output_path

def convert_to_vector(flood_map, metadata, threshold=0.5, min_area=100):
    """Convert the flood map to vector format (GeoJSON)."""
    # Create binary mask
    mask = flood_map >= threshold
    
    # Extract shapes with rasterio
    results = []
    for geom, val in shapes(mask.astype(np.uint8), mask=mask, transform=metadata['transform']):
        if val == 1:  # Only keep flood areas
            geom = shape(geom)
            if geom.area >= min_area:  # Filter by minimum area
                results.append({
                    'geometry': geom,
                    'properties': {'class': 'flood', 'area': geom.area}
                })
    
    # Create GeoDataFrame
    if results:
        gdf = gpd.GeoDataFrame.from_features(results, crs=metadata['crs'])
        return gdf
    else:
        return None

def create_interactive_map(gdf, background_image, metadata, output_path):
    """Create an interactive Folium map."""
    if gdf is None or len(gdf) == 0:
        print("No flood areas to display on interactive map")
        return None
    
    # Reproject to Web Mercator if needed
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    
    # Get center of the data
    center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    
    # Create map
    m = folium.Map(location=center, zoom_start=10, tiles='OpenStreetMap')
    
    # Add flood areas
    folium.GeoJson(
        gdf,
        name='Flood Areas',
        style_function=lambda x: {
            'fillColor': 'blue',
            'color': 'blue',
            'weight': 1,
            'fillOpacity': 0.5
        },
        tooltip=folium.GeoJsonTooltip(fields=['area'], aliases=['Area'])
    ).add_to(m)
    
    # Add heatmap of flood probabilities
    points = []
    for _, row in gdf.iterrows():
        # Sample points from polygons
        for x, y in row.geometry.exterior.coords:
            points.append([y, x, row['area']])
    
    if points:
        HeatMap(points).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save the map
    m.save(output_path)
    print(f"Saved interactive map to {output_path}")
    return output_path

def predict_flood_map(model, image_path, output_dir, patch_size, overlap, threshold):
    """Generate flood prediction for a single image."""
    # Extract filename
    filename = os.path.basename(image_path)
    base_filename = os.path.splitext(filename)[0]
    
    # Create output subdirectory
    image_output_dir = os.path.join(output_dir, base_filename)
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Read the image
    image, metadata = read_geotiff(image_path)
    if image is None:
        print(f"Skipping {filename} due to read error")
        return None
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Generate patches
    patches, positions = generate_patches(processed_image, patch_size, overlap)
    
    # Reshape for model input
    input_patches = np.expand_dims(patches, -1)
    
    # Predict
    print(f"Running prediction on {len(patches)} patches for {filename}")
    predictions = model.predict(input_patches, batch_size=8, verbose=1)
    
    # Stitch predictions
    stitched = stitch_predictions(predictions, positions, image.shape, patch_size, overlap)
    
    # Create binary flood map
    flood_map = create_flood_map(stitched, threshold)
    
    # Save results
    result_paths = {}
    
    # Save prediction probability map
    prob_path = os.path.join(image_output_dir, f"{base_filename}_flood_prob.tif")
    result_paths['probability'] = save_geotiff(stitched, metadata, prob_path)
    
    # Save binary flood map
    map_path = os.path.join(image_output_dir, f"{base_filename}_flood_map.tif")
    result_paths['flood_map'] = save_geotiff(flood_map, metadata, map_path)
    
    # Create visualization
    vis_path = os.path.join(image_output_dir, f"{base_filename}_visualization.png")
    result_paths['visualization'] = create_colormap_visualization(image, flood_map, vis_path)
    
    # Convert to vector format
    gdf = convert_to_vector(flood_map, metadata, threshold)
    if gdf is not None:
        # Save as GeoJSON
        geojson_path = os.path.join(image_output_dir, f"{base_filename}_flood_areas.geojson")
        gdf.to_file(geojson_path, driver='GeoJSON')
        result_paths['geojson'] = geojson_path
        
        # Create interactive map
        map_html_path = os.path.join(image_output_dir, f"{base_filename}_interactive_map.html")
        result_paths['interactive_map'] = create_interactive_map(gdf, image, metadata, map_html_path)
    
    return result_paths

def main():
    """Main function for generating flood prediction maps."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path)
    if model is None:
        return
    
    # List image files
    image_files = list_image_files(args.data_dir)
    if not image_files:
        print(f"No images found in {args.data_dir}")
        return
    
    # Process each image
    results = []
    for image_path in tqdm(image_files, desc="Processing images"):
        result_paths = predict_flood_map(
            model, 
            image_path, 
            args.output_dir, 
            args.patch_size, 
            args.overlap, 
            args.water_threshold
        )
        
        if result_paths:
            result = {
                'image_path': image_path,
                'filename': os.path.basename(image_path),
                'processed_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                **result_paths
            }
            results.append(result)
    
    # Save processing summary
    if results:
        summary_df = pd.DataFrame(results)
        summary_path = os.path.join(args.output_dir, 'prediction_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved prediction summary to {summary_path}")
    
    print(f"Flood map generation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 