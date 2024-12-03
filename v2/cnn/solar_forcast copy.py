# solar_forecast.py

import os
import re
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib

# -----------------------------
# 1. Configuration
# -----------------------------

# Paths
PICS_FOLDER = 'pics'  # Folder containing images
CSV_FILE = 'weather_data.csv'  # CSV file with Lux data
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

RESULTS_DIR = 'results'  # Directory to save results
os.makedirs(RESULTS_DIR, exist_ok=True)

# Image parameters
IMAGE_SIZE = (299, 299)  # Resize images to 299x299
IMAGE_CHANNELS = 3  # RGB

# Prediction horizon
PREDICT_MINUTES_AHEAD = 5

# -----------------------------
# 2. Helper Functions
# -----------------------------

def parse_timestamp_from_filename(filename):
    """
    Extracts datetime from filename formatted as yyyymmddhhmmss.jpg
    """
    basename = os.path.basename(filename)
    name, ext = os.path.splitext(basename)
    try:
        timestamp = datetime.strptime(name, '%Y%m%d%H%M%S')
        return timestamp
    except ValueError:
        return None

def load_images(pics_folder):
    """
    Loads images and their corresponding timestamps.
    Returns a DataFrame with columns ['timestamp', 'image']
    """
    data = []
    for file in os.listdir(pics_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            filepath = os.path.join(pics_folder, file)
            timestamp = parse_timestamp_from_filename(file)
            if timestamp:
                try:
                    img = Image.open(filepath).convert('RGB')
                    img = img.resize(IMAGE_SIZE)
                    img_array = np.array(img) / 255.0  # Normalize to [0,1]
                    data.append({'timestamp': timestamp, 'image': img_array})
                except Exception as e:
                    print(f"Error loading image {file}: {e}")
    df_images = pd.DataFrame(data)
    return df_images

def load_Lux_data(csv_file):
    """
    Loads Lux data from CSV.
    Expects a 'datetime' column in yyyymmddhhmmss format and
    'Global CMP22 (vent/cor) [W/m^2]' column.
    Returns a DataFrame with columns ['timestamp', 'Lux']
    """
    df = pd.read_csv(csv_file)
    # Ensure 'datetime' and 'Global CMP22 (vent/cor) [W/m^2]' columns exist
    if 'datetime' not in df.columns or 'Global CMP22 (vent/cor) [W/m^2]' not in df.columns:
        raise ValueError("CSV must contain 'datetime' and 'Global CMP22 (vent/cor) [W/m^2]' columns.")
    
    # Parse 'datetime' to datetime objects
    df['timestamp'] = pd.to_datetime(df['datetime'].astype(str), format='%Y%m%d%H%M%S', errors='coerce')
    df = df.dropna(subset=['timestamp'])  # Drop rows with invalid timestamps
    
    # Select relevant columns
    df = df[['timestamp', 'Global CMP22 (vent/cor) [W/m^2]']].rename(columns={
        'Global CMP22 (vent/cor) [W/m^2]': 'Lux'
    })
    
    return df

def align_data(df_images, df_Lux, predict_minutes=5):
    """
    Aligns images with Lux data based on timestamp.
    Returns aligned DataFrame with columns ['timestamp', 'image', 'target']
    """
    # Merge on timestamp
    df_merged = pd.merge(df_images, df_Lux, on='timestamp', how='inner')
    
    # Sort by timestamp
    df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)
    
    # Shift Lux to get target 5 minutes ahead
    df_merged['target'] = df_merged['Lux'].shift(-predict_minutes)
    
    # Drop rows with NaN targets
    df_final = df_merged.dropna(subset=['target']).reset_index(drop=True)
    
    return df_final[['timestamp', 'image', 'target']]  # Include timestamp

# -----------------------------
# 3. Data Loading and Preprocessing
# -----------------------------

print("Loading images...")
df_images = load_images(PICS_FOLDER)
print(f"Loaded {len(df_images)} images.")

print("Loading Lux data...")
df_Lux = load_Lux_data(CSV_FILE)
print(f"Loaded Lux data with {len(df_Lux)} entries.")

print("Aligning data...")
df_aligned = align_data(df_images, df_Lux, predict_minutes=PREDICT_MINUTES_AHEAD)
print(f"Aligned dataset has {len(df_aligned)} samples.")

# Convert to numpy arrays
X = np.stack(df_aligned['image'].values)  # Shape: (num_samples, 299, 299, 3)
y = df_aligned['target'].values  # Shape: (num_samples,)
timestamps = df_aligned['timestamp'].values  # Store timestamps

# Reshape y for scaling
y = y.reshape(-1, 1)

# Scale target variable
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

# -----------------------------
# 4. Model Definition
# -----------------------------

def build_cnn_model(input_shape):
    """
    Builds a simple CNN model for regression.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='linear')  # Output layer for regression
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    return model

input_shape = X.shape[1:]  # (299, 299, 3)
model = build_cnn_model(input_shape)
model.summary()

# -----------------------------
# 5. Training
# -----------------------------

# Early stopping to prevent overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

print("Training the model...")
history = model.fit(
    X, y_scaled,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# 6. Results Organization
# -----------------------------

# Create results directory with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = os.path.join(RESULTS_DIR, f'run_{timestamp}')
os.makedirs(run_dir, exist_ok=True)
print(f"Results will be saved in: {run_dir}")

# -----------------------------
# 7. Save Model and Scaler
# -----------------------------

# Save the trained model
model_path = os.path.join(run_dir, 'solar_forecast_cnn_model.keras')
model.save(model_path)
print(f"Model saved to '{model_path}'.")

# Save the scaler
scaler_path = os.path.join(run_dir, 'scaler.pkl')
joblib.dump(scaler_y, scaler_path)
print(f"Scaler saved to '{scaler_path}'.")

print("Training completed successfully.")