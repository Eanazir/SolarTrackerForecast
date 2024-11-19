import os
import pandas as pd
import cv2
import numpy as np
from datetime import timedelta

# Constants
CSV_FILE = "combined_data.csv"
IMAGE_FOLDER = "training_images"
OUTPUT_FOLDER = "output"  # Folder to save filtered data/images
START_DATE = "20230101"  # Specify the start date (YYYYMMDD)
DAYS = 60  # Number of days to extract

# Ensure the output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def filter_week_data(csv_file, start_date, days):
    """
    Filters a week's worth of data from the CSV file.
    """
    # Read the CSV file
    data = pd.read_csv(csv_file)
    
    # Convert the datetime column to a pandas datetime object
    data['datetime'] = pd.to_datetime(data['datetime'], format='%Y%m%d%H%M%S')
    
    # Define the time range
    start_datetime = pd.to_datetime(start_date, format='%Y%m%d')
    end_datetime = start_datetime + timedelta(days=days)

    # Include all data points within the specified range
    filtered_data = data[(data['datetime'] >= start_datetime) & (data['datetime'] <= end_datetime)]
    return filtered_data

def match_images_to_data(filtered_data, image_folder):
    """
    Matches images with the filtered data based on the datetime column.
    Retrieves all data points within a 10-minute window for each image.
    """
    # Create a dictionary for quick lookup of images
    image_files = sorted(os.listdir(image_folder))
    image_dict = {os.path.splitext(img)[0]: img for img in image_files}
    
    matched_images = []
    matched_data = []

    for image_time_str in image_dict.keys():
        # Convert image filename timestamp to datetime
        image_time = pd.to_datetime(image_time_str, format='%Y%m%d%H%M%S')

        # Define the 10-minute window
        start_time = image_time - timedelta(minutes=10)
        end_time = image_time + timedelta(minutes=10)

        # Find all data points within this 10-minute window
        rows_in_window = filtered_data[
            (filtered_data['datetime'] >= start_time) & (filtered_data['datetime'] <= end_time)
        ]

        if not rows_in_window.empty:
            # Add image file path for each matched data point
            matched_images.extend([os.path.join(image_folder, image_dict[image_time_str])] * len(rows_in_window))
            matched_data.append(rows_in_window)
    
    # Concatenate all matched rows into a single DataFrame
    matched_data_df = pd.concat(matched_data).drop_duplicates().sort_values(by='datetime')
    return matched_data_df, matched_images

def save_filtered_data(filtered_data, matched_images):
    """
    Saves the filtered data and copies associated images to the output folder.
    """
    # Convert datetime column back to original string format
    filtered_data['datetime'] = filtered_data['datetime'].dt.strftime('%Y%m%d%H%M%S')
    
    # Save the filtered data to a new CSV file
    filtered_data.to_csv(os.path.join(OUTPUT_FOLDER, "filtered_data.csv"), index=False)
    
    # Save the matched images
    image_output_folder = os.path.join(OUTPUT_FOLDER, "images")
    os.makedirs(image_output_folder, exist_ok=True)
    
    for image_path in set(matched_images):  # Avoid duplicating image saving
        image_name = os.path.basename(image_path)
        img = cv2.imread(image_path)
        if img is not None:
            cv2.imwrite(os.path.join(image_output_folder, image_name), img)
    
    print(f"Filtered data saved to {os.path.join(OUTPUT_FOLDER, 'filtered_data.csv')}")
    print(f"Matched images saved to {image_output_folder}")

# Main script
if __name__ == "__main__":
    # Step 1: Filter the data for the specified week
    filtered_data = filter_week_data(CSV_FILE, START_DATE, DAYS)
    print(f"Filtered {len(filtered_data)} rows from {CSV_FILE}.")
    
    # Step 2: Match images to the filtered data
    matched_data, matched_images = match_images_to_data(filtered_data, IMAGE_FOLDER)
    print(f"Matched {len(matched_images)} images with the filtered data.")
    
    # Step 3: Save the filtered data and matched images
    save_filtered_data(matched_data, matched_images)