# Import necessary libraries
""" Importing Libraries """
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Set up the GPU configuration
""" Configuring GPU Settings """
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Define image dimensions
""" Defining Input Image Shape """
input_image_shape = (500, 500, 3)  # Width, height, and color channels

# Define the base directory relative to the script's current location
""" Setting Up Directory Paths """
base_dir = os.path.join(os.path.dirname(__file__), 'Project 2 Data', 'Data')

# Set up paths for train, validation (valid), and test directories
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'valid')  # Updated to 'valid'
test_dir = os.path.join(base_dir, 'test')

# Confirm paths
print(f"Training directory exists: {os.path.isdir(train_dir)}")
print(f"Validation directory exists: {os.path.isdir(validation_dir)}")
print(f"Test directory exists: {os.path.isdir(test_dir)}")

# Data augmentation settings
""" Defining Data Augmentation for Training and Validation Data """
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

""" Initializing Data Generators with image_dataset_from_directory """
# Initialize data generators
""" Initializing Data Generators """
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(500, 500),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(500, 500),
    batch_size=32,
    class_mode='binary'
)


train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels='inferred',            # Automatically infers labels based on folder names
    label_mode='categorical',      # Suitable for multi-class classification
    batch_size=32,
    image_size=(500, 500),         # Resizes images to (500, 500)
    shuffle=True                   # Shuffle the training data
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    validation_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(500, 500),
    shuffle=False                  # Typically, validation data is not shuffled
)

# Print dataset info to confirm setup
print("Train dataset:", train_dataset)
print("Validation dataset:", validation_dataset)

# Define the CNN model architecture
model = Sequential([
    # First convolutional layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3)),  # 32 filters, 3x3 kernel size
    MaxPooling2D((2, 2)),  # 2x2 pool size
    
    # Second convolutional layer
    Conv2D(64, (3, 3), activation='relu'),  # 64 filters, 3x3 kernel size
    MaxPooling2D((2, 2)),  # 2x2 pool size
    
    # Third convolutional layer
    Conv2D(128, (3, 3), activation='relu'),  # 128 filters, 3x3 kernel size
    MaxPooling2D((2, 2)),  # 2x2 pool size
    
    # Flatten layer to prepare for dense layers
    Flatten(),
    
    # Fully connected (Dense) layer
    Dense(512, activation='relu'),
    Dropout(0.5),  # Dropout layer to reduce overfitting
    
    # Final output layer with 3 neurons (for 3 classes) and softmax activation for multi-class classification
    Dense(3, activation='softmax')
])

# Print model summary to confirm architecture
model.summary()