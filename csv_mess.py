# import pandas as pd

# # Read the CSV file
# df = pd.read_csv('day_data.csv')

# # Define input and output datetime formats
# input_format = '%m/%d/%y %H:%M'
# output_format = '%Y%m%d%H%M%S'

# # Convert the 'datetime' column
# df['datetime'] = pd.to_datetime(df['datetime'], format=input_format).dt.strftime(output_format)

# # Save the updated DataFrame to a new CSV file
# df.to_csv('day_data_converted.csv', index=False)

# print('Datetime conversion complete. Saved to day_data_converted.csv.')


import os
from datetime import datetime

def rename_images():
    # Directory containing the images
    pics_dir = 'pics'
    
    # Get all files in the directory
    for filename in os.listdir(pics_dir):
        try:
            if filename.endswith('.jpg'):
                old_path = os.path.join(pics_dir, filename)
                
                if len(filename) == 18 and filename[:14].isdigit():
                    # Already formatted files - just set seconds to 00
                    new_filename = filename[:12] + '00.jpg'
                else:
                    # Unformatted files - convert from original format
                    datetime_str = filename.split('_image')[0]
                    dt = datetime.strptime(datetime_str, '%Y-%m-%dT%H-%M-%S-%fZ')
                    new_filename = dt.strftime('%Y%m%d%H%M') + '00.jpg'
                
                new_path = os.path.join(pics_dir, new_filename)
                if old_path != new_path:
                    os.rename(old_path, new_path)
                    print(f'Renamed: {filename} -> {new_filename}')
                
        except Exception as e:
            print(f'Error processing {filename}: {str(e)}')

if __name__ == '__main__':
    rename_images()

