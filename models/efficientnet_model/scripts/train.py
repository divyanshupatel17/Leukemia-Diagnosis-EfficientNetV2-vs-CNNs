import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models.efficientnet_model.scripts.model_architecture import create_model
from models.efficientnet_model.scripts.preprocess import load_and_preprocess_data
from models.efficientnet_model.config import BATCH_SIZE, EPOCHS, MODEL_PATH, LEARNING_RATE, IMAGE_SIZE, CLASS_NAMES, DATA_DIR

def plot_training_history(history, save_path_loss, save_path_accuracy):
    """Plots the training history (loss and accuracy) and saves them."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path_loss), exist_ok=True)
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training vs. Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training vs. Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path_loss)
    plt.close()

def train_model(X_train, y_train, X_val, y_val, model, class_weights, batch_size=BATCH_SIZE, epochs=EPOCHS, model_path=MODEL_PATH, learning_rate=LEARNING_RATE):
    """Trains the EfficientNetV2-S model."""

    # Ensure the model directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True)

    # Define optimizer with learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[early_stopping, model_checkpoint],
                        class_weight=class_weights)

    plot_training_history(history, 'models/efficientnet_model/results/figures/training_history.png', 'models/efficientnet_model/results/figures/training_accuracy.png')
    print(f"Model trained and saved to: {model_path}")
    return history

def main():
    """Main function to be called from other modules."""
    # 1. Load Data
    print("Starting EfficientNet training script...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()

    print("Data loaded. X_train shape:", X_train.shape, "y_train shape:", y_train.shape)

    # 2. Create and Compile the Model (Pass image_size to the create_model function)
    model = create_model(image_size=IMAGE_SIZE)
    model.summary()

    # 3. Calculate Class Weights
    class_counts = np.sum(y_train, axis=0)
    total = np.sum(class_counts)
    class_weights = {i: total / (len(CLASS_NAMES) * class_counts[i]) for i in range(len(CLASS_NAMES))}
    print("Class Weights:", class_weights)

    # 4. Train the Model
    history = train_model(X_train, y_train, X_val, y_val, model, class_weights)

    print("Training complete.")
    return 0

if __name__ == "__main__":
    main()




# OUTPUT

# PS D:\VIT_class\4_semester\AI\Project\FINAL_MODEL\model_2_EfficientNetV2_s_cnn\Leukemia_Detection_CNN> python -m scripts.train   
# 2025-03-09 12:59:49.221519: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-03-09 12:59:56.116521: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# Starting training script...
# Found classes: ['Benign', 'Early', 'Pre', 'Pro']
# Loading images from: data/ALL_dataset\Original\Benign
# Loading Benign: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 504/504 [00:01<00:00, 400.32it/s] 
# Loading images from: data/ALL_dataset\Original\Early
# Loading Early: 100%|███████████████████████████████████████████████████████████████████████████████████████████████KeyboardInterrupt
# PS D:\VIT_class\4_semester\AI\Project\FINAL_MODEL\model_2_EfficientNetV2_s_cnn\Leukemia_Detection_CNN> python -m scripts.train
# 2025-03-09 12:59:49.221519: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-03-09 12:59:56.116521: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# Starting training script...
# Found classes: ['Benign', 'Early', 'Pre', 'Pro']
# Loading images from: data/ALL_dataset\Original\Benign
# Loading Benign: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 504/504 [00:01<00:00, 400.32it/s]ebuild TensorFlow with th
# Loading images from: data/ALL_dataset\Original\Early
# Loading Early: 100%|███████████████████████████████████████████████████████████████████████████████████████████████Loading Benign: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 504/504 [00:01<00:00, 400.32it/s]
# Loading images from: data/ALL_dataset\Original\Early
# Loading Early: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 985/985 [00:02<00:00, 422.20it/s]
# Loading images from: data/ALL_dataset\Original\Pre
# Loading Pre: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 963/963 [00:02<00:00, 427.62it/s]
# Loading images from: data/ALL_dataset\Original\Pro
# Loading Pro: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 804/804 [00:01<00:00, 425.17it/s]
# Training data shape: (2083, 224, 224, 3), (2083, 4)
# Validation data shape: (521, 224, 224, 3), (521, 4)
# Testing data shape: (652, 224, 224, 3), (652, 4)
# Data loaded.  X_train shape: (2083, 224, 224, 3) y_train shape: (2083, 4)
# 2025-03-09 13:00:33.032008: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ efficientnetv2-s (Functional)        │ (None, 7, 7, 1280)          │      20,331,360 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ global_average_pooling2d             │ (None, 1280)                │               0 │
# │ (GlobalAveragePooling2D)             │                             │                 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense (Dense)                        │ (None, 128)                 │         163,968 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dropout (Dropout)                    │ (None, 128)                 │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 4)                   │             516 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 20,495,844 (78.19 MB)
#  Trainable params: 20,341,972 (77.60 MB)
#  Non-trainable params: 153,872 (601.06 KB)
# Class Weights: {0: np.float64(1.6122291021671826), 1: np.float64(0.8265873015873015), 2: np.float64(0.8453733766233766), 3: np.float64(1.0131322957198443)}
# Epoch 1/20
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 0s 5s/step - accuracy: 0.6505 - loss: 0.8728WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 444s 6s/step - accuracy: 0.6530 - loss: 0.8675 - val_accuracy: 0.3148 - val_loss: 1.4341
# Epoch 2/20
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 0s 6s/step - accuracy: 0.9617 - loss: 0.1195WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 385s 6s/step - accuracy: 0.9618 - loss: 0.1191 - val_accuracy: 0.9962 - val_loss: 0.0203
# Epoch 3/20
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 377s 6s/step - accuracy: 0.9882 - loss: 0.0463 - val_accuracy: 0.8311 - val_loss: 0.6446
# Epoch 4/20
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 346s 5s/step - accuracy: 0.9825 - loss: 0.0620 - val_accuracy: 0.9962 - val_loss: 0.0230
# Epoch 5/20
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 336s 5s/step - accuracy: 0.9916 - loss: 0.0405 - val_accuracy: 0.9942 - val_loss: 0.0339
# Epoch 6/20
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 0s 5s/step - accuracy: 0.9915 - loss: 0.0302WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 357s 5s/step - accuracy: 0.9915 - loss: 0.0301 - val_accuracy: 0.9981 - val_loss: 0.0114
# Epoch 7/20
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 362s 5s/step - accuracy: 0.9993 - loss: 0.0096 - val_accuracy: 0.9866 - val_loss: 0.0597
# Epoch 8/20
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 364s 6s/step - accuracy: 0.9939 - loss: 0.0188 - val_accuracy: 0.9981 - val_loss: 0.0135
# Epoch 9/20
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 360s 5s/step - accuracy: 0.9982 - loss: 0.0081 - val_accuracy: 0.9885 - val_loss: 0.0438
# Epoch 10/20
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 348s 5s/step - accuracy: 0.9965 - loss: 0.0123 - val_accuracy: 0.9942 - val_loss: 0.0253
# Epoch 11/20
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 358s 5s/step - accuracy: 0.9953 - loss: 0.0110 - val_accuracy: 0.9750 - val_loss: 0.0834
# Epoch 12/20
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 364s 6s/step - accuracy: 0.9948 - loss: 0.0242 - val_accuracy: 0.9789 - val_loss: 0.0751
# Epoch 13/20
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 353s 5s/step - accuracy: 0.9951 - loss: 0.0184 - val_accuracy: 0.9981 - val_loss: 0.0053
# Epoch 14/20
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 360s 5s/step - accuracy: 0.9966 - loss: 0.0173 - val_accuracy: 0.9962 - val_loss: 0.0124
# Epoch 15/20
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 364s 6s/step - accuracy: 0.9965 - loss: 0.0193 - val_accuracy: 0.9750 - val_loss: 0.0624
# Epoch 16/20
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 357s 5s/step - accuracy: 0.9962 - loss: 0.0087 - val_accuracy: 0.9962 - val_loss: 0.0118
# Epoch 17/20
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 345s 5s/step - accuracy: 0.9995 - loss: 0.0031 - val_accuracy: 0.9962 - val_loss: 0.0112
# Epoch 18/20
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 357s 5s/step - accuracy: 0.9987 - loss: 0.0109 - val_accuracy: 0.9789 - val_loss: 0.0666
# Epoch 19/20
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 353s 5s/step - accuracy: 0.9939 - loss: 0.0248 - val_accuracy: 0.9962 - val_loss: 0.0169
# Epoch 20/20
# 66/66 ━━━━━━━━━━━━━━━━━━━━ 322s 5s/step - accuracy: 0.9981 - loss: 0.0095 - val_accuracy: 0.9962 - val_loss: 0.0286
# Model trained and saved to: models/efficientnetv2s_leukemia_model.h5
# Training complete.