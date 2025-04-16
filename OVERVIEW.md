# Flood Prediction and Mapping Using Satellite Imagery

This project uses computer vision techniques to process satellite imagery for predicting and mapping flood-prone areas. The system analyzes historical and current satellite data from NASA's MODIS Near Real-Time Global Flood Mapping Project.

## Project Overview

Flooding is one of the most common and devastating natural disasters worldwide. Early detection and accurate mapping of flood-prone areas are crucial for effective disaster management, risk assessment, and mitigation planning. This project leverages satellite imagery and computer vision techniques to create an automated system for flood prediction and mapping.

## Technical Approach

### 1. Data Acquisition
- Source: NASA's MODIS Near Real-Time Global Flood Mapping Project (http://floodmap.modaps.eosdis.nasa.gov/)
- The system downloads satellite imagery covering the specified time period and region
- Multiple product types are supported (1-day, 2-day, 3-day, and 14-day composites)

### 2. Image Preprocessing
- Reading and interpreting GeoTIFF files with spatial metadata
- Cloud and shadow masking to filter out invalid data
- Image enhancement for better feature extraction
- Creation of training datasets as image patches

### 3. Model Architecture
- U-Net architecture for semantic segmentation
- Encoder-decoder structure with skip connections
- Customized for satellite imagery analysis
- Optimized for water/flood area detection

### 4. Training and Validation
- Dataset split into training, validation, and test sets
- Custom loss function combining binary cross-entropy and Dice coefficient
- Early stopping and learning rate reduction to prevent overfitting
- Performance evaluation using pixel-wise accuracy and Dice coefficient

### 5. Flood Prediction and Mapping
- Processing new satellite imagery through the trained model
- Generating probability maps of flood areas
- Converting predictions to binary flood maps
- Creating vector data (GeoJSON) for integration with GIS systems

### 6. Visualization and Analysis
- Interactive maps showing flood-prone areas
- Time-series analysis of flooding patterns
- Heatmaps showing flood probability

## Components

### Data Acquisition Module
- `src/data_acquisition/download_modis_data.py`: Downloads satellite imagery from NASA's MODIS flood mapping service

### Preprocessing Module
- `src/preprocessing/preprocess_images.py`: Processes raw satellite imagery for training and prediction

### Model Module
- `src/models/train_model.py`: Creates and trains the U-Net model for flood detection

### Visualization Module
- `src/visualization/generate_flood_maps.py`: Generates flood prediction maps and visualizations

### Demo and Utilities
- `demo.py`: Demonstrates the system with synthetic data
- `main.py`: Runs the entire pipeline from data download to visualization

## Usage Examples

### Run the Demo
```
python demo.py
```

### Run the Full Pipeline
```
python main.py --start_date 2023-01-01 --end_date 2023-01-31 --region global
```

### Data Download Only
```
python src/data_acquisition/download_modis_data.py --start_date 2023-01-01 --end_date 2023-01-31 --region asia
```

### Run Predictions with Existing Model
```
python src/visualization/generate_flood_maps.py --model_path models/flood_unet_final_model.h5 --data_dir data/modis/2D2OT --output_dir outputs
```

## Applications

This system can be used for:

1. **Disaster Management**: Quickly identifying areas affected by flooding for emergency response
2. **Urban Planning**: Identifying flood-prone areas for better infrastructure planning
3. **Insurance Risk Assessment**: Evaluating flood risk for property insurance
4. **Climate Change Studies**: Analyzing changes in flooding patterns over time
5. **Agricultural Planning**: Predicting potential crop losses due to flooding

## Future Work

- Integration with additional satellite data sources (Sentinel, Landsat)
- Temporal analysis to detect changing flood patterns over time
- Incorporating elevation data for improved prediction accuracy
- Deployment as a web service for real-time flood monitoring
- Mobile application for field workers and emergency responders

## References

1. NASA MODIS Near Real-Time Global Flood Mapping: http://floodmap.modaps.eosdis.nasa.gov/
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In Medical Image Computing and Computer-Assisted Intervention (MICCAI).
3. DeVries, B., Huang, C., Lang, M. W., Jones, J. W., Huang, W., Creed, I. F., & Carroll, M. L. (2017). Automated Quantification of Surface Water Inundation in Wetlands Using Optical Satellite Imagery. Remote Sensing, 9(8), 807. 