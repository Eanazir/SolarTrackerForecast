# scaler.py

import joblib
import json

# Load MinMaxScaler
scaler = joblib.load('/Users/Eyad/Desktop/Classes/24/fall_24/CSCE_483/SolarTrackerForecast/results/run_20241120_182136/scaler.pkl')

# Extract MinMaxScaler parameters
scaler_params = {
    'min_': scaler.min_.tolist(),
    'scale_': scaler.scale_.tolist(),
    'data_min_': scaler.data_min_.tolist(),
    'data_max_': scaler.data_max_.tolist(),
    'data_range_': scaler.data_range_.tolist()
}

# Save to JSON
with open('scaler_params.json', 'w') as f:
    json.dump(scaler_params, f, indent=4)

print("Scaler parameters saved to scaler_params.json")