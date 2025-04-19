# scripts/train.py
import numpy as np
import os
import matplotlib.pyplot as plt
from models.cnn_model.models.model_architecture import create_model
from models.cnn_model.scripts.preprocess import load_and_preprocess_data, save_data
from models.cnn_model.config import BATCH_SIZE, EPOCHS, MODEL_PATH, LEARNING_RATE, MODEL_DATA_DIR
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import pandas as pd  # For creating tables

def load_data(data_dir, prefix):
    """Loads the preprocessed data"""
    X = np.load(os.path.join(data_dir, f'{prefix}_X.npy'))
    y = np.load(os.path.join(data_dir, f'{prefix}_y.npy'))
    return X, y

def plot_training_history(history, save_path_loss, save_path_accuracy):
    """
    Plots the training history (loss and accuracy) and saves them.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path_loss), exist_ok=True)
    os.makedirs(os.path.dirname(save_path_accuracy), exist_ok=True)
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path_loss)  # Save loss plot
    plt.close() # Close the plot to free memory

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path_accuracy)  # Save accuracy plot
    plt.close()  # Close the plot

def train_model(X_train, y_train, X_val, y_val, model, batch_size=BATCH_SIZE, epochs=EPOCHS, model_path=MODEL_PATH, learning_rate=LEARNING_RATE):
    """
    Trains the CNN model.

    Args:
        X_train (numpy.ndarray): Training data.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Validation data.
        y_val (numpy.ndarray): Validation labels.
        model (tf.keras.models.Sequential): The CNN model.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        model_path (str): Path to save the trained model.
    """

    # Ensure the model directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # Stop if validation loss doesn't improve
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True) # Remove save_format

    # Define optimizer with learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[early_stopping, model_checkpoint])

    # Plot training history
    plot_training_history(history, 'models/cnn_model/results/training_plots/loss_plot.png', 'models/cnn_model/results/training_plots/accuracy_plot.png') # Modified path
    print(f"Model trained and saved to: {model_path}")
    return history

def main():
    """Main function that can be imported and called by other modules."""
    # Create results directories if they don't exist
    os.makedirs("models/cnn_model/results/training_plots", exist_ok=True)
    
    # Check if model directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # 1. Load Data
    augmented_train_dir = os.path.join(MODEL_DATA_DIR, "augmented/train")
    processed_train_dir = os.path.join(MODEL_DATA_DIR, "processed/train")
    
    # Check if preprocessed data exists
    if not os.path.exists(augmented_train_dir) or not os.path.exists(processed_train_dir):
        print("Preprocessed data not found. Running preprocessing first...")
        try:
            from models.cnn_model.scripts.preprocess import main as preprocess_main
            preprocess_main()
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return 1
    
    print("Starting CNN training script...")
    try:
        X_train, y_train = load_data(augmented_train_dir, "train")
        X_val, y_val = load_data(processed_train_dir, "val")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
        
    print("Data loaded. X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
    
    # 2. Create and Compile the Model
    model = create_model()
    model.summary()
    
    # 3. Train the Model
    try:
        history = train_model(X_train, y_train, X_val, y_val, model)
    except Exception as e:
        print(f"Error during training: {e}")
        return 1

    print("Training complete.")
    return 0

if __name__ == "__main__":
    # 1. Load Data
    print("Starting training script...")
    X_train, y_train = load_data(os.path.join(MODEL_DATA_DIR, "augmented/train"), "train")
    X_val, y_val = load_data(os.path.join(MODEL_DATA_DIR, "processed/train"), "val")
    print("Data loaded.  X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
    # 2. Create and Compile the Model
    model = create_model()
    model.summary()
    # 3. Train the Model
    history = train_model(X_train, y_train, X_val, y_val, model)

    print("Training complete.")



# TERMINAL OUTPUT:
# PS D:\VIT_class\4_semester\AI\Project\temp\model_v2\Leukemia_Detection_CNN> python -m scripts.train  
# 2025-03-09 09:58:50.166684: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-03-09 09:58:56.898170: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# Starting training script...
# Data loaded.  X_train shape: (4166, 128, 128, 3) y_train shape: (4166, 4)
# C:\Users\divya\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\keras\src\layers\convolutional\base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
#   super().__init__(activity_regularizer=activity_regularizer, **kwargs)
# 2025-03-09 09:59:24.409152: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ conv2d (Conv2D)                      │ (None, 128, 128, 32)        │             896 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d (MaxPooling2D)         │ (None, 64, 64, 32)          │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_1 (Conv2D)                    │ (None, 64, 64, 64)          │          18,496 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d_1 (MaxPooling2D)       │ (None, 32, 32, 64)          │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_2 (Conv2D)                    │ (None, 32, 32, 128)         │          73,856 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d_2 (MaxPooling2D)       │ (None, 16, 16, 128)         │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ flatten (Flatten)                    │ (None, 32768)               │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense (Dense)                        │ (None, 128)                 │       4,194,432 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dropout (Dropout)                    │ (None, 128)                 │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 4)                   │             516 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 4,288,196 (16.36 MB)
#  Trainable params: 4,288,196 (16.36 MB)
#  Non-trainable params: 0 (0.00 B)
# Epoch 1/20
# 131/131 ━━━━━━━━━━━━━━━━━━━━ 0s 110ms/step - accuracy: 0.5641 - loss: 1.0007WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
# 131/131 ━━━━━━━━━━━━━━━━━━━━ 16s 116ms/step - accuracy: 0.5650 - loss: 0.9988 - val_accuracy: 0.7447 - val_loss: 0.5357
# Epoch 2/20
# 130/131 ━━━━━━━━━━━━━━━━━━━━ 0s 123ms/step - accuracy: 0.8060 - loss: 0.4658WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
# 131/131 ━━━━━━━━━━━━━━━━━━━━ 17s 128ms/step - accuracy: 0.8063 - loss: 0.4654 - val_accuracy: 0.8503 - val_loss: 0.3754
# Epoch 3/20
# 131/131 ━━━━━━━━━━━━━━━━━━━━ 16s 119ms/step - accuracy: 0.8661 - loss: 0.3235 - val_accuracy: 0.8215 - val_loss: 0.4478
# Epoch 4/20
# 131/131 ━━━━━━━━━━━━━━━━━━━━ 0s 111ms/step - accuracy: 0.9073 - loss: 0.2473WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
# 131/131 ━━━━━━━━━━━━━━━━━━━━ 15s 116ms/step - accuracy: 0.9073 - loss: 0.2472 - val_accuracy: 0.8887 - val_loss: 0.3276
# Epoch 5/20
# 130/131 ━━━━━━━━━━━━━━━━━━━━ 0s 116ms/step - accuracy: 0.9321 - loss: 0.1932WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
# 131/131 ━━━━━━━━━━━━━━━━━━━━ 16s 120ms/step - accuracy: 0.9322 - loss: 0.1930 - val_accuracy: 0.8925 - val_loss: 0.2901
# Epoch 6/20
# 130/131 ━━━━━━━━━━━━━━━━━━━━ 0s 114ms/step - accuracy: 0.9504 - loss: 0.1417WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
# 131/131 ━━━━━━━━━━━━━━━━━━━━ 16s 119ms/step - accuracy: 0.9504 - loss: 0.1415 - val_accuracy: 0.8964 - val_loss: 0.3322
# Epoch 7/20
# 131/131 ━━━━━━━━━━━━━━━━━━━━ 16s 121ms/step - accuracy: 0.9571 - loss: 0.1375 - val_accuracy: 0.8887 - val_loss: 0.3281
# Epoch 8/20
# 131/131 ━━━━━━━━━━━━━━━━━━━━ 15s 111ms/step - accuracy: 0.9615 - loss: 0.1382 - val_accuracy: 0.8676 - val_loss: 0.3501
# Epoch 9/20
# 131/131 ━━━━━━━━━━━━━━━━━━━━ 15s 112ms/step - accuracy: 0.9737 - loss: 0.0769 - val_accuracy: 0.8906 - val_loss: 0.3928
# Epoch 10/20
# 130/131 ━━━━━━━━━━━━━━━━━━━━ 0s 112ms/step - accuracy: 0.9588 - loss: 0.1364WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
# 131/131 ━━━━━━━━━━━━━━━━━━━━ 15s 116ms/step - accuracy: 0.9587 - loss: 0.1363 - val_accuracy: 0.8983 - val_loss: 0.3197
# Epoch 11/20
# 131/131 ━━━━━━━━━━━━━━━━━━━━ 16s 119ms/step - accuracy: 0.9820 - loss: 0.0739 - val_accuracy: 0.8983 - val_loss: 0.3124
# Epoch 12/20
# 131/131 ━━━━━━━━━━━━━━━━━━━━ 15s 118ms/step - accuracy: 0.9860 - loss: 0.0349 - val_accuracy: 0.8714 - val_loss: 0.5002
# Epoch 13/20
# 130/131 ━━━━━━━━━━━━━━━━━━━━ 0s 114ms/step - accuracy: 0.9473 - loss: 0.1515WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
# 131/131 ━━━━━━━━━━━━━━━━━━━━ 16s 118ms/step - accuracy: 0.9477 - loss: 0.1504 - val_accuracy: 0.9002 - val_loss: 0.3537
# Epoch 14/20
# 130/131 ━━━━━━━━━━━━━━━━━━━━ 0s 115ms/step - accuracy: 0.9790 - loss: 0.0583WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
# 131/131 ━━━━━━━━━━━━━━━━━━━━ 16s 120ms/step - accuracy: 0.9790 - loss: 0.0585 - val_accuracy: 0.9040 - val_loss: 0.3153
# Epoch 15/20
# 131/131 ━━━━━━━━━━━━━━━━━━━━ 16s 123ms/step - accuracy: 0.9884 - loss: 0.0363 - val_accuracy: 0.9040 - val_loss: 0.3067
# Model trained and saved to: models/saved_models/leukemia_model.h5
# Training complete.