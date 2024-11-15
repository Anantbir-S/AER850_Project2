import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the saved model from the specified location
model_path = r'C:\Users\Anant\Documents\GitHub\AER850_Project2\saved_model101.keras'
model = tf.keras.models.load_model(model_path)

# Define test image paths
test_dir = r'C:\Users\Anant\Documents\GitHub\AER850_Project2\Project 2 Data\Data\test'
test_images = {
    'crack': os.path.join(test_dir, 'crack', 'test_crack.jpg'),
    'missing_head': os.path.join(test_dir, 'missing-head', 'test_missinghead.jpg'),
    'paint_off': os.path.join(test_dir, 'paint-off', 'test_paintoff.jpg')
}

# Preprocess and predict on each test image
for defect, img_path in test_images.items():
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(500, 500))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]

    # Display the image and prediction
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediction: {defect.capitalize()}, Class: {predicted_class}, Confidence: {confidence:.2f}")
    plt.show()
