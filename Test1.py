# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Set up the GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


""" Defined the input image shape """
# Define image dimensions
input_image_shape = (500, 500, 3)  # Width, height, and color channels

# Define the base directory relative to this script's location
base_dir = os.path.join(os.path.dirname(__file__), 'Project 2 Data', 'Data')

# Set up paths for train, validation, and test directories
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

# Verify the paths
print(f"Train directory exists: {os.path.isdir(train_dir)}")
print(f"Validation directory exists: {os.path.isdir(validation_dir)}")
print(f"Test directory exists: {os.path.isdir(test_dir)}")


# Initialize the model with an Input layer
model = Sequential([
    Input(shape=input_image_shape),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Assuming binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

""" Establishing the data """
# Define the base directory relative to the script's current location
base_dir = os.path.join(os.path.dirname(__file__), 'Project 2 Data', 'Data')

# Define subdirectories for train, validation, and test
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Print to verify the paths
print(f"Training directory: {train_dir}")
print(f"Validation directory: {validation_dir}")
print(f"Test directory: {test_dir}")


""" Data Augmentation """

# Define data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,       # Re-scale images to [0,1]
    shear_range=0.2,         # Shear intensity (0.2 is a commonly used value)
    zoom_range=0.2,          # Randomly zoom images
    horizontal_flip=True      # Randomly flip images horizontally
)

# Only re-scale for the validation set (no augmentation)
validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

# Define the directory paths for training and validation sets
train_dir = 'relative/path/to/train'  # Update with the actual relative path
validation_dir = 'relative/path/to/validation'  # Update with the actual relative path

# Flow images in batches from the train directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(500, 500),    # Resize images to (500, 500)
    batch_size=32,             # Adjust batch size as necessary
    class_mode='binary'        # Use 'binary' for binary classification
)

# Flow images in batches from the validation directory
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(500, 500),
    batch_size=32,
    class_mode='binary'
)