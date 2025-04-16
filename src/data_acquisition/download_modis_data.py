#!/usr/bin/env python
"""
Download MODIS flood mapping data from NASA's MODIS Near Real-Time Global Flood Mapping.
Data source: http://floodmap.modaps.eosdis.nasa.gov/
"""

import os
import argparse
import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import time

BASE_URL = "http://floodmap.modaps.eosdis.nasa.gov/downloadData.php"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download MODIS flood mapping data')
    parser.add_argument('--start_date', type=str, default=(datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d'),
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, default=datetime.datetime.now().strftime('%Y-%m-%d'),
                        help='End date in YYYY-MM-DD format')
    parser.add_argument('--region', type=str, default='global',
                        help='Region of interest (e.g., global, asia, africa)')
    parser.add_argument('--product_type', type=str, default='2D2OT',
                        choices=['1D1OS', '2D2OT', '3D3OT', '14DXT'],
                        help='MODIS flood product type')
    return parser.parse_args()

def get_available_data(start_date, end_date, region, product_type):
    """Query available data from the MODIS Flood Mapping site."""
    start_date_str = datetime.datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y.%m.%d')
    end_date_str = datetime.datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y.%m.%d')
    
    params = {
        'product': product_type,
        'platform': 'combined',
        'startDate': start_date_str,
        'endDate': end_date_str,
        'region': region
    }
    
    print(f"Querying available data from {start_date} to {end_date} for region '{region}'...")
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code != 200:
        print(f"Error fetching data: HTTP {response.status_code}")
        return []
    
    # Parse the HTML response to extract download links
    soup = BeautifulSoup(response.text, 'html.parser')
    links = []
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if href.endswith('.tif') and product_type in href:
            links.append(href)
    
    print(f"Found {len(links)} datasets available for download")
    return links

def download_file(url, output_dir):
    """Download a file from URL to the specified output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.basename(url)
    output_path = os.path.join(output_dir, filename)
    
    # Skip if file already exists
    if os.path.exists(output_path):
        print(f"File {filename} already exists, skipping...")
        return output_path
    
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return output_path
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        # Remove partially downloaded file
        if os.path.exists(output_path):
            os.remove(output_path)
        return None

def main():
    """Main function to download MODIS flood data."""
    args = parse_arguments()
    
    # Create output directories
    product_dir = os.path.join(DATA_DIR, "modis", args.product_type)
    if not os.path.exists(product_dir):
        os.makedirs(product_dir)
    
    # Get available data
    data_links = get_available_data(
        args.start_date, 
        args.end_date, 
        args.region, 
        args.product_type
    )
    
    if not data_links:
        print("No data found for the specified parameters")
        return
    
    # Download data
    print(f"Downloading {len(data_links)} files to {product_dir}...")
    
    download_results = []
    for link in data_links:
        # Construct full URL if it's a relative path
        if not link.startswith('http'):
            full_url = f"http://floodmap.modaps.eosdis.nasa.gov/{link}"
        else:
            full_url = link
        
        output_path = download_file(full_url, product_dir)
        if output_path:
            download_date = datetime.datetime.now().strftime('%Y-%m-%d')
            filename = os.path.basename(output_path)
            download_results.append({
                'filename': filename,
                'url': full_url,
                'download_date': download_date,
                'file_path': output_path
            })
        
        # Be nice to the server
        time.sleep(1)
    
    # Save download metadata
    if download_results:
        metadata_file = os.path.join(product_dir, "download_metadata.csv")
        df = pd.DataFrame(download_results)
        
        if os.path.exists(metadata_file):
            existing_df = pd.read_csv(metadata_file)
            df = pd.concat([existing_df, df]).drop_duplicates(subset=['filename']).reset_index(drop=True)
        
        df.to_csv(metadata_file, index=False)
        print(f"Downloaded {len(download_results)} files successfully. Metadata saved to {metadata_file}")
    else:
        print("No files were downloaded successfully.")

if __name__ == "__main__":
    main() 