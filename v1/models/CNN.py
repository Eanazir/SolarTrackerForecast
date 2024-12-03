# models/cnn_model.py

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model

def create_cnn_model(input_shape=(128, 128, 3)):
    # Input for images
    img_input = Input(shape=input_shape, name='Image_Input')

    # CNN layers
    # Adds multiple convolutional layers with ReLU activation and padding.
	# Each convolutional layer is followed by a max-pooling layer to reduce spatial dimensions.
    x = Conv2D(32, (3,3), activation='relu', padding='same')(img_input)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    
    # Flatten and add dense layers
    # Flattens the output of the convolutional layers.
	# Adds a dense layer to produce a feature vector.
    x = Flatten()(x)
    img_output = Dense(128, activation='relu')(x)

    # Define model
    cnn_model = Model(inputs=img_input, outputs=img_output, name='CNN_Model')
    return cnn_model