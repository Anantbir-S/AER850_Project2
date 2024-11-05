# Import necessary libraries
""" Importing Libraries """
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

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

# Define updated hyperparameters in model architecture
""" Defining Updated Model Hyperparameters """
updated_hyperparameters_model = Sequential([
    # First convolutional layer with LeakyReLU activation
    Conv2D(32, (3, 3), activation='linear', input_shape=input_image_shape),  # Start with 'linear' activation
    LeakyReLU(alpha=0.1),  # Adding LeakyReLU activation after the layer
    MaxPooling2D((2, 2)),
    
    # Second convolutional layer with ReLU activation
    Conv2D(64, (3, 3), activation='relu'),  # Using 'relu' as another option
    MaxPooling2D((2, 2)),
    
    # Third convolutional layer with LeakyReLU activation
    Conv2D(128, (3, 3), activation='linear'),
    LeakyReLU(alpha=0.1),
    MaxPooling2D((2, 2)),
    
    # Flatten layer to prepare for dense layers
    Flatten(),
    
    # Fully connected (Dense) layer with ReLU activation
    Dense(512, activation='relu'),  # You can try 'elu' as an alternative
    Dropout(0.5),
    
    # Final output layer with 3 neurons for 3 classes, using softmax activation
    Dense(3, activation='softmax')
])

# Compile the updated model
updated_hyperparameters_model.compile(
    optimizer=Adam(),  # Adam optimizer
    loss='categorical_crossentropy',  # Suitable for multi-class classification
    metrics=['accuracy']  # Track accuracy during training
)

# Print model summary to confirm architecture
print("\nUpdated Model Architecture with Hyperparameters:")
updated_hyperparameters_model.summary()

# Train the model and capture training history
history = updated_hyperparameters_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10  # You can adjust the number of epochs based on your needs
)

# Plot training & validation accuracy values
plt.figure(figsize=(14, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')

plt.tight_layout()
plt.show()