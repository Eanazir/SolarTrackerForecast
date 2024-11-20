import boto3
import datetime
import os
import pandas as pd
from urllib.parse import urlparse

# AWS configuration
aws_region = 'us-east-2'
s3_bucket = 'solar-tracker-images'
aws_access_key_id = 'AKIA3FLDXONAOOY3MHOO'
aws_secret_access_key = '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'

# Initialize S3 client
s3 = boto3.client(
    's3',
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Read CSV file
csv_path = '/Users/Eyad/Desktop/Classes/24/fall_24/CSCE_483/SolarTrackerForecast/weather_data_2024-11-16.csv'
df = pd.read_csv(csv_path)

# Create pics directory if it doesn't exist
if not os.path.exists('pics'):
    os.makedirs('pics')

# Process each row with URL and datetime
for index, row in df.iterrows():
    try:
        # Parse S3 URL to get key
        url = row['image_url']
        parsed_url = urlparse(url)
        key = parsed_url.path.lstrip('/')
        
        # Use exact datetime value from CSV
        filename = str(row['datetime']) + '.jpg'
        
        # Create local filename path
        local_filename = os.path.join('pics', filename)
        
        # Download file
        print(f'Downloading {local_filename}...')
        s3.download_file(s3_bucket, key, local_filename)
        
    except Exception as e:
        print(f'Error downloading {url}: {str(e)}')

print('Download complete.')