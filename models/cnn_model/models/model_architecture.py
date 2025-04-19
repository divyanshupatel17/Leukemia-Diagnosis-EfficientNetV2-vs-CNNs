# models/model_architecture.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from models.cnn_model.config import IMAGE_SIZE, NUM_CLASSES

def create_model():
    """
    Creates a CNN model for leukemia detection.

    Returns:
        tf.keras.models.Sequential: The compiled CNN model.
    """
    model = Sequential()

    # Convolutional Block 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), padding='same')) # Assuming color images (3 channels)
    model.add(MaxPooling2D((2, 2)))

    # Convolutional Block 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # Convolutional Block 3 (Optional - add more blocks for deeper networks)
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))


    # Flatten
    model.add(Flatten())

    # Dense Layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5)) # Regularization

    # Output Layer
    model.add(Dense(4, activation='softmax'))  # Softmax for multi-class

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # Use categorical_crossentropy
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # Example Usage (for testing the model definition)
    model = create_model()
    model.summary() # Print a summary of the model's layers