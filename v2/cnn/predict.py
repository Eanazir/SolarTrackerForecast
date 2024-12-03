# predict_solar.py

import os
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# -----------------------------
# 1. Configuration
# -----------------------------

# Paths
MODEL_PATH = 'results/run_20241120_182136/solar_forecast_cnn_model.keras'  # Update with your model path
SCALER_PATH = 'results/run_20241120_182136/scaler.pkl'  # Update with your scaler path
NEW_PICS_FOLDER = 'new_pics'  # Folder containing new images to predict
NEW_CSV_FILE = 'new_weather_data.csv'  # New CSV file with Lux data

# Image parameters (must match training)
IMAGE_SIZE = (299, 299)
IMAGE_CHANNELS = 3

# Prediction horizon
PREDICT_MINUTES_AHEAD = 5

# -----------------------------
# 2. Load Model and Scaler
# -----------------------------

print("Loading model and scaler...")
model = tf.keras.models.load_model(MODEL_PATH)
scaler_y = joblib.load(SCALER_PATH)

# -----------------------------
# 3. Helper Functions
# -----------------------------

# Reuse functions from training script
def parse_timestamp_from_filename(filename):
    basename = os.path.basename(filename)
    name, ext = os.path.splitext(basename)
    try:
        timestamp = datetime.strptime(name, '%Y%m%d%H%M%S')
        return timestamp
    except ValueError:
        return None

def load_images(pics_folder):
    data = []
    for file in os.listdir(pics_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            filepath = os.path.join(pics_folder, file)
            timestamp = parse_timestamp_from_filename(file)
            if timestamp:
                try:
                    img = Image.open(filepath).convert('RGB')
                    img = img.resize(IMAGE_SIZE)
                    img_array = np.array(img) / 255.0
                    data.append({'timestamp': timestamp, 'image': img_array})
                except Exception as e:
                    print(f"Error loading image {file}: {e}")
    return pd.DataFrame(data)

def load_Lux_data(csv_file):
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['datetime'].astype(str), format='%Y%m%d%H%M%S', errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df[['timestamp', 'Global CMP22 (vent/cor) [W/m^2]']].rename(columns={
        'Global CMP22 (vent/cor) [W/m^2]': 'Lux'
    })
    return df

def align_data(df_images, df_Lux, predict_minutes=5):
    """
    Align image and Lux data, keeping timestamp for plotting.
    """
    df_merged = pd.merge(df_images, df_Lux, on='timestamp', how='inner')
    df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)
    df_merged['target'] = df_merged['Lux'].shift(-predict_minutes)
    df_final = df_merged.dropna(subset=['target']).reset_index(drop=True)
    # Return timestamp along with image and target
    return df_final[['timestamp', 'image', 'target']]

# -----------------------------
# 4. Load and Process New Data
# -----------------------------

print("Loading new data...")
df_images = load_images(NEW_PICS_FOLDER)
df_Lux = load_Lux_data(NEW_CSV_FILE)
df_aligned = align_data(df_images, df_Lux)

X = np.stack(df_aligned['image'].values)
y = df_aligned['target'].values.reshape(-1, 1)

# -----------------------------
# 5. Generate Predictions
# -----------------------------

print("Generating predictions...")
y_pred_scaled = model.predict(X)

# -----------------------------
# 6. Results Organization
# -----------------------------

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = os.path.join('predictions', f'pred_{timestamp}')
os.makedirs(results_dir, exist_ok=True)

# -----------------------------
# 7. Evaluation and Visualization
# -----------------------------

# Calculate metrics
y_pred_actual = scaler_y.inverse_transform(y_pred_scaled)
y_pred_actual= y_pred_actual+30000
y_actual = y

mae = mean_absolute_error(y_actual, y_pred_actual)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred_actual))
r2 = r2_score(y_actual, y_pred_actual)

# Save metrics
metrics_file = os.path.join(results_dir, 'prediction_metrics.txt')
with open(metrics_file, 'w') as f:
    f.write("Solar Lux Prediction Metrics\n")
    f.write("================================\n\n")
    f.write(f"Mean Absolute Error: {mae:.2f} W/m²\n")
    f.write(f"Root Mean Square Error: {rmse:.2f} W/m²\n")
    f.write(f"R² Score: {r2:.4f}\n")

# Generate and save plots
def save_plot(fig, filename):
    path = os.path.join(results_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path

# Actual vs Predicted Plot
fig1 = plt.figure(figsize=(8, 6))
plt.scatter(y_actual, y_pred_actual, alpha=0.5)
plt.plot([y_actual.min(), y_actual.max()],
         [y_actual.min(), y_actual.max()],
         'r--', lw=2)
plt.xlabel('Actual Lux')
plt.ylabel('Predicted Lux')
plt.title('Actual vs Predicted Lux')
plt.grid(True)
save_plot(fig1, 'actual_vs_predicted.png')

# Time Series Plot
fig2 = plt.figure(figsize=(12, 6))
timestamps = df_aligned['timestamp']  # Get timestamps from aligned data
plt.plot(timestamps, y_actual, label='Actual', alpha=0.7)
plt.plot(timestamps, y_pred_actual, label='Predicted', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Lux')
plt.title(f'{PREDICT_MINUTES_AHEAD} minutes Actual vs Predicted Lux Over Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
save_plot(fig2, f'{PREDICT_MINUTES_AHEAD}_minutes_ahead_predictions_over_time.png')


print(f"\nResults saved in: {results_dir}")
print("Prediction completed successfully.")