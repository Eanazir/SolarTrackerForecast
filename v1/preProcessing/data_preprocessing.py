# preprocessing/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import joblib

def preprocess_meteorological_data(file_path):
    # Load data
    met_data = pd.read_csv(file_path)

    # Combine DATE and MST into a single datetime column
    # met_data['Timestamp'] = pd.to_datetime(met_data['DATE'] + ' ' + met_data['MST'])
    # Combine DATE and MST into a single datetime column
    met_data['Timestamp'] = pd.to_datetime(met_data['datetime'].astype(str), format='%Y%m%d%H%M%S')

    # Sort data by Timestamp
    met_data.sort_values('Timestamp', inplace=True)

    # Handle missing values in input features using KNN imputation
    input_features = [
        'Tower Dry Bulb Temp [deg C]', 'Tower RH [%]', 'Station Pressure [mBar]',
        'Avg Wind Speed @ 6ft [m/s]', 'Avg Wind Direction @ 6ft [deg from N]'
    ]

    # Initialize KNN imputer
    imputer = KNNImputer(n_neighbors=5)

    # Fit and transform the input features
    met_data_imputed = imputer.fit_transform(met_data[input_features])

    # Update the DataFrame with imputed values
    met_data[input_features] = met_data_imputed

    # Handle missing values in the target variable separately
    target_variable = 'Global CMP22 (vent/cor) [W/m^2]'

    # Optionally interpolate missing target values
    # met_data[target_variable].interpolate(method='time', inplace=True)
    
    # Option 2: Drop rows with missing target values (uncomment if preferred)
    met_data.dropna(subset=[target_variable], inplace=True)

    # Rename columns for simplicity
    met_data.rename(columns={
        'Tower Dry Bulb Temp [deg C]': 'Temperature',
        'Tower RH [%]': 'Humidity',
        'Station Pressure [mBar]': 'Pressure',
        'Avg Wind Speed @ 6ft [m/s]': 'Wind Speed',
        'Avg Wind Direction @ 6ft [deg from N]': 'Wind Direction',
        'Global CMP22 (vent/cor) [W/m^2]': 'Irradiance'
    }, inplace=True)

    # Feature scaling for input features
    scaler = MinMaxScaler()
    met_data[['Temperature', 'Humidity', 'Pressure', 'Wind Speed']] = scaler.fit_transform(
        met_data[['Temperature', 'Humidity', 'Pressure', 'Wind Speed']]
    )
    joblib.dump(scaler, 'scaler_y.pkl')

    # Wind Direction encoding (convert degrees to sine and cosine components)
    met_data['Wind Dir Sin'] = np.sin(np.deg2rad(met_data['Wind Direction']))
    met_data['Wind Dir Cos'] = np.cos(np.deg2rad(met_data['Wind Direction']))
    met_data.drop('Wind Direction', axis=1, inplace=True)

    # Temporal features
    met_data['Hour'] = met_data['Timestamp'].dt.hour / 23.0  # Normalize Hour
    met_data['DayOfYear'] = met_data['Timestamp'].dt.dayofyear / 365.0  # Normalize DayOfYear

    # Prepare target variables (future irradiance)
    target = 'Irradiance'
    for minutes in [5, 15, 30, 60]:
        met_data[f'Irradiance_{minutes}min_ahead'] = met_data[target].shift(-minutes)

    # **Removed the line that drops rows with NaN values after shifting**
    # We will handle dropping NaN values after merging with images
    # met_data.dropna(inplace=True)

    # Reset index and return the processed DataFrame
    return met_data.reset_index(drop=True)