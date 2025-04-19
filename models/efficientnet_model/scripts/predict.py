import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
from models.efficientnet_model.config import MODEL_PATH, IMAGE_SIZE, CLASS_NAMES, DATA_DIR

def preprocess_image(image_path, target_size=IMAGE_SIZE):
    """
    Loads, resizes, and preprocesses a single image for prediction.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_image(image_array, model_path=MODEL_PATH):
    """
    Predicts the class of a single preprocessed image.

    Args:
        image_array: A numpy array (already preprocessed).

    Returns:
        A tuple: (predicted_class, confidence) or (None, None) on error.
    """
    try:
        model = load_model(model_path)
        # Make the prediction
        prediction = model.predict(np.expand_dims(image_array, axis=0))
        predicted_class_index = np.argmax(prediction)
        confidence = prediction[0][predicted_class_index]
        predicted_class = CLASS_NAMES[predicted_class_index]
        return predicted_class, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

def main():
    """Main function to be called from other modules."""
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return 1
        
    # Example usage
    example_image_path = os.path.join(DATA_DIR, "Original/Benign/image_0001.jpg")  # Replace with your image path
    
    if not os.path.exists(example_image_path):
        print(f"Error: Test image not found at {example_image_path}")
        # Try to find an alternative test image
        benign_dir = os.path.join(DATA_DIR, "Original/Benign")
        if os.path.exists(benign_dir):
            image_files = [f for f in os.listdir(benign_dir) if f.endswith('.jpg') or f.endswith('.png')]
            if image_files:
                example_image_path = os.path.join(benign_dir, image_files[0])
                print(f"Using alternative image: {example_image_path}")
            else:
                print("No image files found in Benign directory")
                return 1
        else:
            print(f"Benign directory not found at {benign_dir}")
            return 1
    
    image_array = preprocess_image(example_image_path)
    
    if image_array is not None:
        predicted_class, confidence = predict_image(image_array)
        if predicted_class:
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.4f}")
        else:
            print("Prediction failed.")
            return 1
    else:
        print("Image preprocessing failed.")
        return 1
    
    print("Prediction complete.")
    return 0

if __name__ == '__main__':
    main()