import json
import os
import pandas as pd
from datetime import datetime
import shutil

def get_timestamp_components(timestamp_str):
    """Extract timestamp components for matching"""
    try:
        dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        return {
            'month': dt.month,
            'day': dt.day,
            'hour': dt.hour,
            'minute': dt.minute
        }
    except:
        try:
            # For image filenames
            dt = datetime.strptime(timestamp_str, 'image_%Y%m%d_%H%M%S.jpg')
            return {
                'month': dt.month,
                'day': dt.day,
                'hour': dt.hour,
                'minute': dt.minute
            }
        except:
            return None

def timestamps_match(ts1, ts2):
    """Check if timestamps match on month, day, hour, and minute"""
    return (ts1['month'] == ts2['month'] and 
            ts1['day'] == ts2['day'] and 
            ts1['hour'] == ts2['hour'] and 
            ts1['minute'] == ts2['minute'])

def main():
    # Load JSON data
    json_data = load_json_data('day_test.json')
    
    # Get list of image files
    image_dir = 'uploaded_images'
    image_files = [f for f in os.listdir(image_dir) if f.startswith('image_')]
    
    # Create output directory
    output_dir = 'renamed_images'
    os.makedirs(output_dir, exist_ok=True)
    
    # Match data and rename images
    matched_data = []
    for image_file in image_files:
        image_ts = get_timestamp_components(image_file)
        if not image_ts:
            continue
            
        # Find matching data point
        for data_point in json_data:
            data_ts = get_timestamp_components(data_point['timestamp'])
            if not data_ts:
                continue
                
            if timestamps_match(image_ts, data_ts):
                # Create new timestamp from data point
                dt = datetime.strptime(data_point['timestamp'], '%Y-%m-%d %H:%M:%S')
                new_timestamp = dt.strftime('%Y%m%d%H%M%S')
                
                # Rename and copy image
                old_path = os.path.join(image_dir, image_file)
                new_path = os.path.join(output_dir, f'{new_timestamp}.jpg')
                shutil.copy2(old_path, new_path)
                
                # Add to matched data
                matched_data.append({
                    'datetime': new_timestamp,
                    'Global CMP22 (vent/cor) [W/m^2]': data_point['ambientWeatherLightLux']
                })
                break
    
    # Create and save CSV
    df = pd.DataFrame(matched_data)
    df.to_csv('weather_data_old.csv', index=False)
    
    print(f"Processed {len(matched_data)} matches")
    print("Created weather_data.csv and renamed images in 'renamed_images' folder")

def load_json_data(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    main()