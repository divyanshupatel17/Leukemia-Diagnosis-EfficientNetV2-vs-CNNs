# scripts/preprocess.py
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical  # For one-hot encoding
from tqdm import tqdm  # For progress bars
from models.cnn_model.config import IMAGE_SIZE, TEST_SIZE, RANDOM_STATE, VALIDATION_SPLIT, DATA_DIR, MODEL_DATA_DIR, CLASS_NAMES  # Import data directories

def load_and_preprocess_data(data_dir=os.path.join(DATA_DIR, "Original"), target_size=IMAGE_SIZE, test_size=TEST_SIZE, random_state=RANDOM_STATE, validation_split=VALIDATION_SPLIT):
    """
    Loads, preprocesses, and splits the image data into training, validation and testing sets.
    The function is modified as it expects the test data in the `.npy` format, while it expects the images for training to be present inside a directory
    """
    X = []  # Image data
    y = []  # Labels (0, 1, 2, 3 corresponding to class names)

    if 'test' in data_dir or 'val' in data_dir:
        # Load from .npy files if in test or val (validation)
        X = np.load(os.path.join(data_dir, 'test_X.npy')) # loads the X data
        y = np.load(os.path.join(data_dir, 'test_y.npy')) # loads the y data
        print('Testing or Validation data loaded using .npy files')
        return X, X, X, y, y, y # returning dummy data to work with the rest of code.
    else: # For train data
        class_names = sorted(os.listdir(data_dir))  # Get class names dynamically.  Assumes subdirectories are class names.
        class_to_index = {class_name: idx for idx, class_name in enumerate(class_names)} # create a dictionary to map each folder to it's index
        print(f"Found classes: {class_names}")

        # Iterate through the class directories
        for class_name in class_names:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):  # Skip if it's not a directory
                continue
            print(f"Loading images from: {class_dir}")
            # Load images and labels
            for filename in tqdm(os.listdir(class_dir), desc=f"Loading {class_name}"): # Progress bar for each class
                if filename.endswith(".jpg") or filename.endswith(".png"):  # Add other image extensions if needed
                    try:
                        img_path = os.path.join(class_dir, filename)
                        img = Image.open(img_path).convert("RGB") # Ensure 3 channels (RGB)
                        img = img.resize(target_size)
                        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                        X.append(img_array)
                        y.append(class_to_index[class_name]) # Use the class index from the dictionary
                    except Exception as e:
                        print(f"Error loading or processing image {img_path}: {e}")
        X = np.array(X)
        y = np.array(y)

        # One-Hot Encode the Labels (for categorical_crossentropy)
        y = to_categorical(y, num_classes=len(class_names))

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y) # Stratify maintains class proportions

        # Split training data into training and validation
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_split, random_state=random_state, stratify=y_train)


        print(f"Training data shape: {X_train.shape}, {y_train.shape}")
        print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
        print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test

def augment_data(X, y, augmentations=None):
        """
        Applies data augmentation to the input data.

        Args:
            X (numpy.ndarray): The image data.
            y (numpy.ndarray): The labels.
            augmentations (list, optional): A list of augmentation functions from tensorflow.keras.preprocessing.image.ImageDataGenerator. Defaults to None.

        Returns:
            tuple: The augmented data (X_augmented, y_augmented).
        """
        if augmentations is None:
            return X, y # Return original if no augmentation

        X_augmented = []
        y_augmented = []
        for i in tqdm(range(len(X)), desc="Augmenting images"):
            img = X[i]
            label = y[i]
            # Apply each augmentation
            for augmentation in augmentations:
                #img_augmented = augmentation.apply_transform(img)  # Apply transform, no 'params' needed anymore  (Old code)
                img_augmented = augmentation.apply_transform(img, transform_parameters={}) # fix for old tensorflow versions
                X_augmented.append(img_augmented)
                y_augmented.append(label)  # Keep the same label for augmented images

        X_augmented = np.array(X_augmented)
        y_augmented = np.array(y_augmented)
        return X_augmented, y_augmented

def create_augmentation_pipeline(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
                                shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'):
    """
    Creates an ImageDataGenerator object for data augmentation.  This is a common and efficient method.
    Args:
        rotation_range (int): Degrees for random rotations.
        width_shift_range (float): Fraction of total width for shifting horizontally.
        height_shift_range (float): Fraction of total height for shifting vertically.
        shear_range (float): Shear angle in counter-clockwise direction.
        zoom_range (float): Range for random zoom.
        horizontal_flip (bool): Whether to randomly flip images horizontally.
        fill_mode (str):  Points outside the boundaries of the input are filled according to the given mode
        (e.g. 'nearest', 'constant', 'reflect', or 'wrap').
    Returns:
        tensorflow.keras.preprocessing.image.ImageDataGenerator: The ImageDataGenerator object.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    return ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode
    )


def save_data(X, y, data_dir, prefix):
    """
    Saves the processed data to the specified directory.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    np.save(os.path.join(data_dir, f'{prefix}_X.npy'), X)
    np.save(os.path.join(data_dir, f'{prefix}_y.npy'), y)

def main():
    """Main function that can be imported and called by other modules."""
    # --- 1. Preprocess ---
    data_dir = os.path.join(DATA_DIR, "Original")  # Path to your original data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(data_dir)

    # --- 2. Save Processed ---
    processed_train_dir = os.path.join(MODEL_DATA_DIR, "processed", "train")
    processed_test_dir = os.path.join(MODEL_DATA_DIR, "processed", "test")
    save_data(X_train, y_train, processed_train_dir, "train")
    save_data(X_val, y_val, processed_train_dir, "val")
    save_data(X_test, y_test, processed_test_dir, "test")

    # --- 3. Augment Training Data  ---
    # Create Augmentation Pipeline
    augmentation_pipeline = create_augmentation_pipeline()

    # Apply Augmentation
    X_train_augmented, y_train_augmented = augment_data(X_train, y_train, augmentations=[augmentation_pipeline])

    # Combine Original and Augmented Data (Important!)
    X_train = np.concatenate((X_train, X_train_augmented), axis=0)
    y_train = np.concatenate((y_train, y_train_augmented), axis=0)

    # Save Augmented Data
    augmented_train_dir = os.path.join(MODEL_DATA_DIR, "augmented", "train")
    save_data(X_train, y_train, augmented_train_dir, "train")

    # Save final datasets to train, validation and test directories
    train_dir = os.path.join(MODEL_DATA_DIR, "train")
    validation_dir = os.path.join(MODEL_DATA_DIR, "validation")
    test_dir = os.path.join(MODEL_DATA_DIR, "test")
    
    save_data(X_train, y_train, train_dir, "train")
    save_data(X_val, y_val, validation_dir, "val")
    save_data(X_test, y_test, test_dir, "test")

    print("Preprocessing, splitting, and augmentation complete.")
    return 0

if __name__ == "__main__":
    main()

# TERMINAL OUTPUT:
# PS D:\AI\Project\Leukemia_Detection_CNN> python -m scripts.preprocess
# 2025-03-02 00:40:15.290573: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-03-02 00:40:21.968920: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# Found classes: ['Benign', 'Early', 'Pre', 'Pro']
# Loading images from: data/archive/Original\Benign
# Loading Benign: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 504/504 [00:04<00:00, 120.66it/s]
# Loading images from: data/archive/Original\Early
# Loading Early: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 985/985 [00:07<00:00, 128.99it/s]
# Loading images from: data/archive/Original\Pre
# Loading Pre: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 963/963 [00:07<00:00, 133.65it/s] 
# Loading images from: data/archive/Original\Pro
# Loading Pro: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 804/804 [00:04<00:00, 193.86it/s] 
# Training data shape: (2083, 128, 128, 3), (2083, 4)
# Validation data shape: (521, 128, 128, 3), (521, 4)
# Testing data shape: (652, 128, 128, 3), (652, 4)
# Augmenting images: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2083/2083 [00:00<00:00, 173613.16it/s]
# Preprocessing, splitting, and augmentation complete.

# Explaination: (What above code have done, what directories are created and what i have to do now)
# Okay, let's analyze the latest preprocess.py output and chart the course ahead.

# 1. What preprocess.py Did

# Image Loading: The script loaded all your images from the data/archive/Original/ directory.

# Preprocessing: Images were resized to 128x128 pixels, and pixel values were normalized to the range [0, 1].

# Splitting: The dataset was split into training, validation, and testing sets.

# Augmentation: Data augmentation was applied to your training data.

# Generated .npy files. Created the .npy files

# data/processed/train/train_X.npy

# data/processed/train/train_y.npy

# data/processed/train/val_X.npy

# data/processed/train/val_y.npy

# data/processed/test/test_X.npy

# data/processed/test/test_y.npy

# data/augmented/train/train_X.npy

# data/augmented/train/train_y.npy

# The script has been executed, and the data has been augmented.

# 2. Next Steps:

# Now that you have created the .npy files with the augmented data, you must train the model.

# Run python -m scripts.train. This script will create and train your CNN model.

# Please run the command, and share the results.