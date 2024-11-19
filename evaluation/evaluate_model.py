# evaluation/evaluate_model.py

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def evaluate_model(model, merged_data):
    # Prepare data (similar to training data preparation)
    sequence_length = 60
    features = ['Temperature', 'Humidity', 'Pressure', 'Wind Speed', 'Wind Dir Sin', 'Wind Dir Cos',
                'Hour', 'DayOfYear']  # Exclude direct current irradiance as a feature

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

    # Use the last 20% for evaluation
    split_index = int(0.8 * len(X_num_seq))
    X_num_test = X_num_seq[split_index:]
    X_img_test = X_img_seq[split_index:]
    y_test = y_seq[split_index:]

    # Predictions
    y_pred = model.predict([X_img_test, X_num_test])

    # Calculate RMSE and MAE for each horizon
    horizons = [5, 15, 30, 60]
    for i, minutes in enumerate(horizons):
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        print(f'{minutes}-Minute Ahead Prediction - RMSE: {rmse:.2f}, MAE: {mae:.2f}')

        # Plot actual vs predicted
        plt.figure(figsize=(10, 4))
        plt.plot(y_test[:, i], label='Actual')
        plt.plot(y_pred[:, i], label='Predicted')
        plt.title(f'{minutes}-Minute Ahead Prediction')
        plt.xlabel('Samples')
        plt.ylabel('Irradiance (W/m^2)')
        plt.legend()
        plt.show()