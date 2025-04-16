#!/usr/bin/env python
"""
Flood Prediction Dataset Analysis Tool
-------------------------------------
This script demonstrates loading and analyzing the flood prediction dataset
for the Flood Vision project.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(SCRIPT_DIR, 'flood_dataset.json')
CSV_PATH = os.path.join(SCRIPT_DIR, 'flood_data.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'img', 'overlays')

def load_data():
    """Load dataset from both JSON and CSV files"""
    # Load JSON data
    with open(JSON_PATH, 'r') as f:
        json_data = json.load(f)
    
    # Load CSV data
    csv_data = pd.read_csv(CSV_PATH)
    
    return json_data, csv_data

def generate_sample_overlay(location_id, save_path):
    """Generate sample overlay visualization for a location"""
    plt.figure(figsize=(10, 6))
    
    # Create a water-like background
    x = np.linspace(0, 10, 1000)
    for i in range(5):
        plt.plot(x, np.sin(x + i/5) * 0.1 + i/10, 
                 color='royalblue', alpha=0.3, linewidth=5)
    
    # Set background color
    plt.gca().set_facecolor('lightcyan')
    
    # Add a colormap for flood risk
    cm = plt.cm.get_cmap('YlOrRd')
    risk_levels = np.linspace(0, 1, 100)
    plt.scatter([5] * 100, [0.8] * 100, c=risk_levels, 
                cmap=cm, alpha=0.0)  # Hidden, just to create colorbar
    cbar = plt.colorbar(orientation='horizontal', label='Flood Risk Level')
    
    # Annotate with location
    plt.title(f"Flood Risk Overlay for Location {location_id}", fontsize=14)
    plt.text(5, 0.5, "FLOOD PREDICTION\nOVERLAY", 
             ha='center', va='center', fontsize=20, 
             color='white', weight='bold',
             bbox=dict(facecolor='royalblue', alpha=0.6, boxstyle='round,pad=0.5'))
    
    # Hide axes
    plt.axis('off')
    
    # Save the overlay
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return save_path

def analyze_flood_risks(data):
    """Analyze flood risks from dataset"""
    # Extract location data
    locations = []
    for location in data['locations']:
        loc_data = {
            'id': location['id'],
            'name': location['name'],
            'lat': location['coordinates']['lat'],
            'lng': location['coordinates']['lng'],
            'risk_factors': location['risk_factors'],
        }
        
        # Get latest data point
        if location['data_points']:
            latest = location['data_points'][0]  # Assuming sorted by date
            loc_data.update({
                'rainfall_mm': latest['rainfall_mm'],
                'river_level_m': latest['river_level_m'],
                'soil_saturation_pct': latest['soil_saturation_pct'],
                'flood_probability': latest['flood_probability'],
                'timestamp': latest['timestamp']
            })
            
            # Generate overlay if needed
            overlay_path = os.path.join(OUTPUT_DIR, f"{location['id'].lower()}_overlay.png")
            if not os.path.exists(overlay_path):
                generate_sample_overlay(location['id'], overlay_path)
                
        locations.append(loc_data)
    
    return pd.DataFrame(locations)

def calculate_risk_score(row):
    """Calculate composite risk score from multiple factors"""
    # Basic formula:
    # 50% probability + 20% rainfall + 15% river level + 15% soil saturation
    probability_factor = row['flood_probability'] * 0.5
    
    # Normalize other factors to 0-100 scale
    rainfall_norm = min(100, row['rainfall_mm'] / 2)  # Cap at 100
    river_norm = min(100, row['river_level_m'] * 20)  # 5m -> 100%
    soil_norm = row['soil_saturation_pct']
    
    rainfall_factor = rainfall_norm * 0.2
    river_factor = river_norm * 0.15
    soil_factor = soil_norm * 0.15
    
    return probability_factor + rainfall_factor + river_factor + soil_factor

def generate_summary_report(csv_data):
    """Generate a summary report of flood risks"""
    # Add risk score
    csv_data['risk_score'] = csv_data.apply(calculate_risk_score, axis=1)
    
    # Sort by risk score
    sorted_data = csv_data.sort_values('risk_score', ascending=False)
    
    print("\n===== FLOOD RISK REPORT =====")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-----------------------------")
    
    for _, row in sorted_data.iterrows():
        risk_category = "HIGH RISK" if row['risk_score'] > 75 else \
                        "MEDIUM RISK" if row['risk_score'] > 50 else "LOW RISK"
        
        print(f"\n{risk_category}: {row['location_name']} ({row['location_id']})")
        print(f"  Risk Score: {row['risk_score']:.1f}/100")
        print(f"  Flood Probability: {row['flood_probability']}%")
        print(f"  Current Conditions:")
        print(f"    - Rainfall: {row['rainfall_mm']} mm")
        print(f"    - River Level: {row['river_level_m']} m")
        print(f"    - Soil Saturation: {row['soil_saturation_pct']}%")
    
    print("\n-----------------------------")
    return sorted_data

def main():
    """Main function to run the analysis"""
    print("Loading flood prediction dataset...")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    json_data, csv_data = load_data()
    
    # Analyze location risk data
    location_analysis = analyze_flood_risks(json_data)
    
    # Generate summary report
    risk_report = generate_summary_report(csv_data)
    
    print("\nAnalysis complete. Overlay images generated in:", OUTPUT_DIR)
    
    return json_data, csv_data, location_analysis, risk_report

if __name__ == "__main__":
    main() 