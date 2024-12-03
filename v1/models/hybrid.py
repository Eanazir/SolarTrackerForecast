# models/hybrid_model.py

from tensorflow.keras.layers import Concatenate, Dense
from tensorflow.keras.models import Model
from models.CNN import create_cnn_model
from models.LSTM import create_lstm_model

def create_hybrid_model(sequence_length, num_features):
    # Create CNN and LSTM models
    cnn_model = create_cnn_model()
    lstm_model = create_lstm_model(sequence_length, num_features)

    # Combine models
    combined = Concatenate()([cnn_model.output, lstm_model.output])

    # Fully connected layers
    # Adds additional dense layers to learn from the combined feature vectors.
    x = Dense(64, activation='relu')(combined)
    x = Dense(32, activation='relu')(x)

    # Output layer
    # Adds a final dense layer with 4 neurons to predict the irradiance at 4 future time intervals.
    output = Dense(4, name='Output')(x)  # Predicting 4 future values

    # Define hybrid model
    hybrid_model = Model(inputs=[cnn_model.input, lstm_model.input], outputs=output, name='Hybrid_Model')
    return hybrid_model