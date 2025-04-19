import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from models.efficientnet_model.config import IMAGE_SIZE, TEST_SIZE, RANDOM_STATE, VALIDATION_SPLIT, DATA_DIR, MODEL_DATA_DIR, CLASS_NAMES

def load_and_preprocess_data(data_dir=os.path.join(DATA_DIR, "Original"), target_size=IMAGE_SIZE, test_size=TEST_SIZE, random_state=RANDOM_STATE, validation_split=VALIDATION_SPLIT):
    """
    Loads, preprocesses, and splits the image data into training, validation, and testing sets.
    """
    X = []  # Image data
    y = []  # Labels (0, 1, 2, 3 corresponding to class names)
    class_to_index = {class_name: idx for idx, class_name in enumerate(CLASS_NAMES)}
    print(f"Found classes: {CLASS_NAMES}")

    # Iterate through the class directories
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        print(f"Loading images from: {class_dir}")
        # Load images and labels
        for filename in tqdm(os.listdir(class_dir), desc=f"Loading {class_name}"):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                try:
                    img_path = os.path.join(class_dir, filename)
                    img = Image.open(img_path).convert("RGB")  # Ensure 3 channels (RGB)
                    img = img.resize(target_size)
                    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                    X.append(img_array)
                    y.append(class_to_index[class_name])  # Use the class index from the dictionary
                except Exception as e:
                    print(f"Error loading or processing image {img_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    # One-Hot Encode the Labels (for categorical_crossentropy)
    y = to_categorical(y, num_classes=len(CLASS_NAMES))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)  # Stratify maintains class proportions

    # Split training data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_split, random_state=random_state, stratify=y_train)

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_data(X, y, data_dir, prefix):
    """
    Saves the processed data to the specified directory.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    np.save(os.path.join(data_dir, f'{prefix}_X.npy'), X)
    np.save(os.path.join(data_dir, f'{prefix}_y.npy'), y)

def preprocess_image(image_path, target_size=IMAGE_SIZE):
    """
    Loads, resizes, and preprocesses a single image for prediction.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        return img_array  # Return as numpy array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def main():
    """Main function to be called from other modules."""
    # Load and preprocess data
    data_dir = os.path.join(DATA_DIR, "Original")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(data_dir)
    
    # Save processed data
    processed_train_dir = os.path.join(MODEL_DATA_DIR, "processed", "train")
    processed_test_dir = os.path.join(MODEL_DATA_DIR, "processed", "test")
    save_data(X_train, y_train, processed_train_dir, "train")
    save_data(X_val, y_val, processed_train_dir, "val")
    save_data(X_test, y_test, processed_test_dir, "test")
    
    # Save final datasets to train, validation and test directories
    train_dir = os.path.join(MODEL_DATA_DIR, "train")
    validation_dir = os.path.join(MODEL_DATA_DIR, "validation")
    test_dir = os.path.join(MODEL_DATA_DIR, "test")
    
    save_data(X_train, y_train, train_dir, "train")
    save_data(X_val, y_val, validation_dir, "val")
    save_data(X_test, y_test, test_dir, "test")
    
    print("Data loaded, preprocessed, and saved.")
    return 0

if __name__ == '__main__':
    main()