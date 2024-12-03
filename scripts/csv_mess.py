import pandas as pd

# Read the CSV file
df = pd.read_csv('new_weather_data.csv')

# Define input and output datetime formats
input_format = '%m/%d/%y %H:%M'
output_format = '%Y%m%d%H%M%S'

# Convert the 'datetime' column
df['datetime'] = pd.to_datetime(df['datetime'], format=input_format).dt.strftime(output_format)

# Save back to the same CSV file
df.to_csv('new_weather_data.csv', index=False)

print('Datetime conversion complete. Updated new_weather_data.csv.')