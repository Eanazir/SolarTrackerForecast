# solar_forecast.py

import os
import re
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
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

# Image parameters
IMAGE_SIZE = (299,299)  # Resize images to 299x299
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
X = np.stack(df_aligned['image'].values)
y = df_aligned['target'].values
timestamps = df_aligned['timestamp'].values  # Store timestamps

# Reshape y for scaling
y = y.reshape(-1, 1)

# Scale target variable
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_scaled, test_size=0.2, random_state=42
)

# Get both training and test indices in one split
train_indices, test_indices = train_test_split(
    range(len(X)), 
    test_size=0.2, 
    random_state=42
)

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


input_shape = X_train.shape[1:]  # (224, 224, 3)
model = build_cnn_model(input_shape)
model.summary()

# -----------------------------
# 5. Training
# -----------------------------

# Early stopping to prevent overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("Training the model...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# 6. Results Organization
# -----------------------------

# Create results directory with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = os.path.join('results', f'run_{timestamp}')
os.makedirs(results_dir, exist_ok=True)

# -----------------------------
# 7. Evaluation and Visualization
# -----------------------------

# Get model predictions
y_pred_scaled = model.predict(X_test)

# Extract test timestamps
test_timestamps = timestamps[test_indices]

# Initialize metrics file
metrics_file = os.path.join(results_dir, 'metrics.txt')
with open(metrics_file, 'w') as f:
    f.write("Solar Lux Prediction Metrics\n")
    f.write("================================\n\n")

# Calculate metrics
y_pred_actual = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

mae = mean_absolute_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
r2 = r2_score(y_test_actual, y_pred_actual)

# Save metrics
with open(metrics_file, 'a') as f:
    f.write(f"Mean Absolute Error: {mae:.2f} W/m²\n")
    f.write(f"Root Mean Square Error: {rmse:.2f} W/m²\n")
    f.write(f"R² Score: {r2:.4f}\n")

# Generate and save plots
def save_plot(fig, filename):
    path = os.path.join(results_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path

# 1. Actual vs Predicted Plot
fig1 = plt.figure(figsize=(8, 6))
plt.scatter(y_test_actual, y_pred_actual, alpha=0.5)
plt.plot([y_test_actual.min(), y_test_actual.max()],
         [y_test_actual.min(), y_test_actual.max()],
         'r--', lw=2)
plt.xlabel('Actual Lux')
plt.ylabel('Predicted Lux')
plt.title('Actual vs Predicted Lux')
plt.grid(True)
save_plot(fig1, 'actual_vs_predicted.png')

# 2. Training History Plot
fig2 = plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training')
plt.plot(history.history['val_mae'], label='Validation')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
save_plot(fig2, 'training_history.png')

# 3. Time Series Plot
fig3 = plt.figure(figsize=(12, 6))

# Sort the data by timestamps for clear visualization
sorted_indices = np.argsort(test_timestamps)
sorted_timestamps = test_timestamps[sorted_indices]
sorted_actual = y_test_actual[sorted_indices]
sorted_predicted = y_pred_actual[sorted_indices]

plt.plot(sorted_timestamps, sorted_actual, 
         label='Actual', alpha=0.7, linewidth=2)
plt.plot(sorted_timestamps, sorted_predicted, 
         label='Predicted', alpha=0.7, linewidth=2)
plt.xlabel('Time')
plt.ylabel('Lux')
plt.title(f'{PREDICT_MINUTES_AHEAD} minutes Actual vs Predicted Lux Over Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

save_plot(fig3, f'{PREDICT_MINUTES_AHEAD} minutes ahead predictions_over_time.png')

# -----------------------------
# 8. Save Model Artifacts
# -----------------------------

# Save model
model_path = os.path.join(results_dir, 'solar_forecast_cnn_model.keras')
model.save(model_path)

# Save scaler
scaler_path = os.path.join(results_dir, 'scaler.pkl')
joblib.dump(scaler_y, scaler_path)

print(f"\nResults saved in: {results_dir}")
print("Script completed successfully.")