# train_model.py

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Enable mixed precision if supported
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Define base directory and paths for train, validation, and test directories
base_dir = os.path.join(os.path.dirname(__file__), 'Project 2 Data', 'Data')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'valid')

# Image dimensions and batch size
input_image_shape = (500, 500, 3)
img_size = (500, 500)
batch_size = 32

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.18,
    rotation_range=26,
    horizontal_flip=True
)

# Rescaling for validation data
valid_datagen = ImageDataGenerator(rescale=1./255)

# Data Generators with consistent target size
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    validation_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the Neural Network Architecture
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_image_shape),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),


    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),


    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),


    layers.Flatten(),
    layers.Dense(128, activation='relu'),  # Increased dense layer size
    layers.BatchNormalization(),


    layers.Dense(3, activation='softmax', dtype='float32')  # Specify float32 for final layer with mixed precision
])

# Compile the model with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks for Early Stopping
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
]

# Train the model
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=valid_generator,
    callbacks=callbacks
)

# Model Evaluation - Plot Accuracy and Loss
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.show()