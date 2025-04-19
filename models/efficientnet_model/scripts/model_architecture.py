import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from models.efficientnet_model.config import IMAGE_SIZE, NUM_CLASSES

def create_model(image_size=IMAGE_SIZE):
    """Creates an EfficientNetV2-S model for leukemia detection using transfer learning."""

    # 1. Load pre-trained EfficientNetV2-S (without the top classification layer)
    base_model = EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=(image_size[0], image_size[1], 3),
    )

    # 2. Unfreeze some layers for fine-tuning
    for layer in base_model.layers[-40:]:  # Unfreeze the last 40 layers
        layer.trainable = True

    # 3. Create the model
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    return model