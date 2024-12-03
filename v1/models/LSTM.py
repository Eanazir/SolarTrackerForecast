import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense
import joblib


def create_sequences(data,sequence_length):
    #creates sequences the minutes we want to predict
    #target is the entry right below the first n sequences. 
    seq, target = [], []
    for i in range(len(data) - sequence_length):
        # Extract input sequence
        seq.append(data[i:i+sequence_length, :-1])  # All columns except target
        # Extract target value (15-minutes later)        
        target.append(data[f'Irradiance_{sequence_length}min_ahead'].iloc[i + sequence_length-1])   # Get 15minLabel value after sequence
    return np.array(seq), np.array(target)


def create_lstm_model1(data, sequence_len):
    #sequence length is the time ahead we want to predict
    
    # Assuming data is already preprocessed and passed as a DataFrame
    X_sequences, y = create_sequences(data.values, sequence_len)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_sequences, y, test_size=0.3, random_state=42)
    
    # Build LSTM model
    n_timesteps, n_features = X_train.shape[1], X_train.shape[2]
    model = Sequential()
    model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features)))
    model.add(LSTM(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, 
            validation_data=(X_test, y_test), verbose=2, shuffle=False)

    # Make predictions
    y_pred = model.predict(X_test)
    scaler_y = joblib.load('scaler_y.pkl')
    # transform back to get the actual value 
    y_pred = scaler_y.inverse_transform(y_pred)
    y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    # Evaluate the model
    mse = tf.keras.losses.MeanSquaredError()
    print(f'Mean Squared Error: {mse(y_test, y_pred).numpy()}')

    return model

def create_lstm_model(sequence_length, num_features):
    # Define LSTM model
    lstm_input = Input(shape=(sequence_length, num_features), name='LSTM_Input')
    x = LSTM(64, activation='relu', return_sequences=True)(lstm_input)
    x = LSTM(32, activation='relu')(x)
    lstm_output = Dense(64, activation='relu')(x)
    lstm_output = Dense(1)(lstm_output)

    # Define model
    lstm_model = Model(inputs=lstm_input, outputs=lstm_output, name='LSTM_Model')
    lstm_model.compile(optimizer='adam', loss='mse')
    return lstm_model


