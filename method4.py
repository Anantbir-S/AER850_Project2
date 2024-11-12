# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:09:56 2024

@author: Anant
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import matplotlib.pyplot as plt
import os

# Define image dimensions
input_image_shape = (500, 500, 3)

# Define base directory
base_dir = os.path.join(os.path.dirname(__file__), 'Project 2 Data', 'Data')

# Set up paths for train, validation (valid), and test directories
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

# Enhanced data augmentation with more transformations to reduce overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,           
    zoom_range=0.2,            
    horizontal_flip=True,
    rotation_range=20,         
    brightness_range=[0.8, 1.2],
    width_shift_range=0.1,     
    height_shift_range=0.1,    
    fill_mode='nearest'
)

# Validation data generator with rescaling only
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create image data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(500, 500),
    batch_size=32,  # Reduced batch size
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(500, 500),
    batch_size=32,
    class_mode='categorical'
)

# Define the CNN model with increased dropout and regularization
model = Sequential([
    Input(shape=input_image_shape),
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.2),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # Second Convolutional Block
    Conv2D(64, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    LeakyReLU(negative_slope=0.2),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # Third Convolutional Block
    Conv2D(128, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    LeakyReLU(negative_slope=0.2),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # Fourth Convolutional Block
    Conv2D(256, (3, 3), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    LeakyReLU(negative_slope=0.2),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # Fully Connected Layers
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.7),  # Higher dropout for fully connected layer

    # Output Layer
    Dense(3, activation='softmax')
])

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return float(lr * tf.math.exp(-0.1))

lr_scheduler = LearningRateScheduler(scheduler)

# Compile the model with a lower initial learning rate
model.compile(optimizer=Adam(learning_rate=0.0003),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Train the model with callbacks
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,  # Increased epochs to allow for gradual training
    callbacks=[lr_scheduler, early_stopping, reduce_lr]
)

# Plot training and validation accuracy/loss
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')

plt.tight_layout()
plt.show()
