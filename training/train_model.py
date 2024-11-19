# training/train_model.py

import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from models.hybrid import create_hybrid_model
import pandas as pd

def train_hybrid_model(met_data, images, image_timestamps):
    # Prepare data
    sequence_length = 60  # Number of past minutes to consider
    features = ['Temperature', 'Humidity', 'Pressure', 'Wind Speed', 'Wind Dir Sin', 'Wind Dir Cos',
                'Hour', 'DayOfYear']

    # Align images with meteorological data
    merged_data = align_data_with_images(met_data, images, image_timestamps)

    # Extract features and targets
    X_num = merged_data[features].values
    y = merged_data[[f'Irradiance_{minutes}min_ahead' for minutes in [5, 15, 30, 60]]].values
    X_img = np.array(merged_data['Image'].tolist())

    # Create sequences
    def create_sequences(X_num, X_img, y, seq_length):
        X_num_seq, X_img_seq, y_seq = [], [], []
        for i in range(len(X_num) - seq_length):
            X_num_seq.append(X_num[i:i+seq_length])
            X_img_seq.append(X_img[i+seq_length-1])  # Use image at last timestamp
            y_seq.append(y[i+seq_length-1])
        return np.array(X_num_seq), np.array(X_img_seq), np.array(y_seq)

    X_num_seq, X_img_seq, y_seq = create_sequences(X_num, X_img, y, sequence_length)

    # Train-test split
    split_index = int(0.8 * len(X_num_seq))
    X_num_train, X_num_test = X_num_seq[:split_index], X_num_seq[split_index:]
    X_img_train, X_img_test = X_img_seq[:split_index], X_img_seq[split_index:]
    y_train, y_test = y_seq[:split_index], y_seq[split_index:]

    # Create model
    num_features = X_num_train.shape[2]
    model = create_hybrid_model(sequence_length, num_features)

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    # Callbacks
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3)

    # Train model
    history = model.fit(
        [X_img_train, X_num_train],
        y_train,
        validation_data=([X_img_test, X_num_test], y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr]
    )

    return model, history, merged_data

def align_data_with_images(met_data, images, image_timestamps):
    # Create a DataFrame for image timestamps
    image_df = pd.DataFrame({'Timestamp': image_timestamps, 'Image': images})
    image_df['Timestamp'] = pd.to_datetime(image_df['Timestamp'])

    # Merge meteorological data with images using an inner join
    met_data['Timestamp'] = pd.to_datetime(met_data['Timestamp'])
    merged_data = pd.merge(met_data, image_df, on='Timestamp', how='inner')

    # Prepare target variables (future irradiance)
    target = 'Irradiance'
    for minutes in [5, 15, 30, 60]:
        merged_data[f'Irradiance_{minutes}min_ahead'] = merged_data[target].shift(-minutes)

    # Drop rows with any remaining missing values (after shifting)
    merged_data.dropna(inplace=True)

    # Reset index
    merged_data.reset_index(drop=True, inplace=True)

    return merged_data