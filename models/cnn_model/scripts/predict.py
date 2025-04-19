# scripts/predict.py
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf
from models.cnn_model.config import MODEL_PATH, IMAGE_SIZE, CLASS_NAMES, DATA_DIR
from tensorflow.keras.preprocessing import image

def preprocess_image(image_path, target_size=IMAGE_SIZE):
    """
    Loads, resizes, and preprocesses a single image for prediction.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

def predict_image(image_path, model_path=MODEL_PATH):
    """
    Predicts the class of a single image.
    """
    # 1. Preprocess the image
    img_array = preprocess_image(image_path)

    # 2. Load the Model
    model = load_model(model_path)

    # 3. Make the Prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    predicted_class = CLASS_NAMES[predicted_class_index]

    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")

    return predicted_class, confidence

def main():
    """Main function to be called from other modules."""
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return 1
        
    # Replace with the path to your test image
    test_image_path = os.path.join(DATA_DIR, "Original/Benign/WBC-Benign-001.jpg")
    
    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}. Please update the test_image_path variable.")
        return 1
    else:
        predict_image(test_image_path)
    print("Prediction complete.")
    return 0

if __name__ == "__main__":
    # Example Usage
    # Replace with the path to your test image (OR Uncomment below any 1 of 4 sample image paths and check if output is correctly showing by our trained model) 
    # test_image_path = os.path.join(DATA_DIR, "Original/Benign/WBC-Benign-001.jpg") # Example path - for 1. Benign
    test_image_path = os.path.join(DATA_DIR, "Original/Early/WBC-Malignant-Early-001.jpg") # Example path - for 2. Early
    # test_image_path = os.path.join(DATA_DIR, "Original/Pre/WBC-Malignant-Pre-001.jpg") # Example path - for 3. Pre
    # test_image_path = os.path.join(DATA_DIR, "Original/Pro/WBC-Malignant-Pro-001.jpg") # Example path - for 4. Pro
    
    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}. Please update the test_image_path variable.")
    else:
        predict_image(test_image_path)
    print("Prediction complete.")



# TERMINAL OUTPUT:
# PS D:\AI\Project\Leukemia_Detection_CNN> python -m scripts.predict   
# 2025-03-02 02:10:59.576384: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-03-02 02:11:00.468827: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-03-02 02:11:03.813354: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 84ms/step
# Predicted Class: Early
# Confidence: 0.9994
# Prediction complete.

# Explaination: (What above code have done, what directories are created and what i have to do now)

# Okay, the output of python -m scripts.predict shows that the prediction script is functioning correctly.
# What predict.py did:

# Model Loading: The script successfully loaded your trained model from models/saved_models/leukemia_model.h5.

# Image Preprocessing: It loaded and preprocessed the input image, resizing and normalizing it, as per the instructions.

# Prediction: The model made a prediction on the preprocessed image.

# Output: The script printed the predicted class and confidence score:

# Predicted Class: Early

# Confidence: 0.9994

# Analysis:

# Successful Prediction: The model has predicted an image to be from Early with high confidence (99.94%).

# What to do next

# Now, you have all the results.

# Include all the plots of the training and validation loss curves.

# Include the confusion matrix.

# Include the tables.

# Write a complete report to finish the task.