# preprocessing/image_preprocessing.py

import cv2
import numpy as np
import glob
import os
import pandas as pd

def preprocess_images(image_folder):
    image_paths = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))
    images = []
    image_timestamps = []

    for path in image_paths:
        # Extract timestamp from image filename
        # Assuming filename format: YYYYMMDDHHMMSS.jpg
        filename = os.path.basename(path)
        timestamp_str = filename.replace('.jpg', '')
        timestamp = pd.to_datetime(timestamp_str, format='%Y%m%d%H%M%S')

        img = cv2.imread(path)
        if img is None:
            continue  # Skip if the image is not readable
        # Iterates over each image path, reads the image using OpenCV, and resizes it to 128x128 pixels
	    # Normalizes pixel values to the range [0, 1] by dividing by 255
        img = cv2.resize(img, (128, 128))
        img = img / 255.0  # Normalize pixel values
        images.append(img)
        image_timestamps.append(timestamp)

    return images, image_timestamps