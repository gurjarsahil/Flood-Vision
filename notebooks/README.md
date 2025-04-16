# Jupyter Notebooks for Flood Prediction

This directory contains Jupyter notebooks that explain and demonstrate the machine learning pipeline used in the Flood Vision project. The notebooks provide step-by-step walkthroughs of the data preprocessing, model training, and evaluation processes.

## Notebook Structure

The notebooks are designed to be followed in sequential order:

1. `data_preprocessing.ipynb`: Preparing satellite imagery and environmental data
2. `model_training.ipynb`: Training the flood prediction and image segmentation models
3. `model_evaluation.ipynb`: Evaluating model performance and interpreting results
4. `inference_demo.ipynb`: Demonstration of making predictions on new data

## Setup Requirements

To run these notebooks, you need:

1. Python 3.7+ with Jupyter installed
2. Required libraries:
   - TensorFlow 2.x
   - Keras
   - OpenCV
   - NumPy
   - Pandas
   - Matplotlib
   - scikit-learn
   - GDAL (for geospatial data processing)
   - Rasterio
   - earthpy
   - xarray

## Data Preprocessing Pipeline (`data_preprocessing.ipynb`)

This notebook covers:

1. **Satellite Image Collection**:
   - Loading Landsat, Sentinel, and MODIS satellite imagery
   - Understanding band combinations and their significance
   - Filtering for cloud cover and quality

2. **Spectral Index Calculation**:
   - Normalized Difference Water Index (NDWI)
   - Modified NDWI (MNDWI)
   - Normalized Difference Vegetation Index (NDVI)
   - Automated Water Extraction Index (AWEI)

3. **Feature Engineering**:
   - Elevation and slope calculation from DEM data
   - Distance to water bodies
   - Soil type and permeability features
   - Land cover classification

4. **Data Integration**:
   - Combining satellite data with weather measurements
   - Aligning historical flood records with environmental variables
   - Creating labeled datasets for training

## Model Training (`model_training.ipynb`)

This notebook demonstrates:

1. **Segmentation Model Training**:
   - U-Net architecture for water body segmentation
   - Transfer learning approaches
   - Data augmentation strategies
   - Training process with callback explanations

2. **Flood Prediction Model Training**:
   - Random Forest for feature importance analysis
   - Gradient Boosting for initial modeling
   - Deep learning approach with temporal data
   - Hyperparameter tuning

3. **Model Architecture Details**:
   - Layer-by-layer explanation of neural networks
   - Activation functions and their roles
   - Loss functions appropriate for flood prediction
   - Regularization techniques to prevent overfitting

## Model Evaluation (`model_evaluation.ipynb`)

This notebook explores:

1. **Performance Metrics**:
   - Accuracy, Precision, Recall, and F1-Score calculation
   - IoU (Intersection over Union) for segmentation evaluation
   - Time-to-event prediction error analysis
   - ROC curves and AUC interpretation

2. **Visualization Techniques**:
   - Confusion matrices
   - Prediction maps with uncertainty visualization
   - Feature importance plots
   - Learning curves

3. **Error Analysis**:
   - Common failure cases
   - Geographical variations in performance
   - Seasonal effects on prediction accuracy
   - Strategies for model improvement

## Inference Demonstration (`inference_demo.ipynb`)

This notebook shows:

1. **End-to-End Prediction Pipeline**:
   - Loading new satellite imagery
   - Preprocessing steps
   - Running inference with trained models
   - Post-processing results

2. **Result Interpretation**:
   - Flood risk probability maps
   - Time-to-flood estimates
   - Area and infrastructure impact assessment
   - Confidence intervals for predictions

3. **Output Formats**:
   - GeoTIFF export for GIS applications
   - Visualization for web interface
   - Report generation for emergency response teams
   - Time-series forecasts

## Running the Notebooks

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Navigate to the notebook you want to run

3. Execute cells in sequence (Shift+Enter)

## Using Your Own Data

To adapt these notebooks for your specific region:

1. Replace the input data paths in the data preprocessing notebook
2. Adjust the geographical coordinates and projection systems
3. Modify the hyperparameters based on your region's characteristics
4. Retrain the models with your local data

## Common Issues and Solutions

- **Memory Errors**: Reduce batch size or use data generators
- **GDAL Installation Problems**: Follow the troubleshooting guide in `docs/gdal_setup.md`
- **GPU Acceleration**: Set the TensorFlow configuration for GPU usage
- **Projection Inconsistencies**: Ensure all geospatial data uses the same coordinate reference system

## Additional Resources

- Detailed documentation in the `docs/` directory
- Sample datasets in `data/samples/`
- Pre-trained models in `models/pretrained/`
- Reference papers in `docs/references/`

## Citations

If you use these notebooks in your research, please cite:

```
Kumar, S. (2023). FloodVision: A Machine Learning Approach for Flood Prediction Using Satellite Imagery. 
DOI: 10.xxxx/xxxxx
``` 