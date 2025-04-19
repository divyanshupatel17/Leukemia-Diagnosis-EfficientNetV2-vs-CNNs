import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import sys
import argparse
sys.path.append('..')
from models.cnn_model.config import MODEL_PATH, IMAGE_SIZE, CLASS_NAMES, DATA_DIR
from PIL import Image

# Define class names if not in config
if 'CLASS_NAMES' not in globals():
    CLASS_NAMES = ["Benign", "Early", "Pre", "Pro"]

def load_and_preprocess_image(img_path, target_size=IMAGE_SIZE):
    """
    Load and preprocess a single image for prediction.
    
    Args:
        img_path (str): Path to the image file.
        target_size (tuple): Target size for resizing.
        
    Returns:
        tuple: Original image and preprocessed image array.
    """
    # Convert to absolute path and normalize path separators
    img_path = os.path.abspath(img_path)
    print(f"Loading image from: {img_path}")
    
    # Check if file exists
    if not os.path.exists(img_path):
        raise ValueError(f"Could not load image from {img_path} - File does not exist")
    
    try:
        # Load image using PIL instead of OpenCV for better compatibility
        img = Image.open(img_path)
        img_rgb = np.array(img.convert('RGB'))
        
        # Resize image
        img_resized = cv2.resize(img_rgb, target_size)
        
        # Normalize pixel values
        img_normalized = img_resized / 255.0
        
        # Expand dimensions to create batch
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_rgb, img_batch
    except Exception as e:
        raise ValueError(f"Error processing image {img_path}: {str(e)}")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Create a Grad-CAM heatmap for a specific prediction index.
    
    Args:
        img_array (numpy.ndarray): Preprocessed image array.
        model (tf.keras.Model): Trained model.
        last_conv_layer_name (str): Name of the last convolutional layer.
        pred_index (int, optional): Index of the prediction to explain.
        
    Returns:
        numpy.ndarray: Grad-CAM heatmap.
    """
    # First, ensure the model has been called at least once
    _ = model(img_array)
    
    # Find the last convolutional layer if not specified
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                print(f"Using last convolutional layer: {last_conv_layer_name}")
                break
    
    # Create a model that maps the input image to the activations of the last conv layer
    # and the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # Gradient of the output neuron with respect to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Vector of mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by the computed gradient values
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def apply_gradcam(image_path, model_path, output_path=None, layer_name='conv2d_2'):
    """
    Apply Grad-CAM to visualize model's focus areas.
    
    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the saved model.
        output_path (str, optional): Path to save the visualization.
        layer_name (str): Name of the last convolutional layer.
        
    Returns:
        numpy.ndarray: Grad-CAM visualization.
    """
    # Load the model
    model = load_model(model_path)
    
    # Load and preprocess the image
    original_img, preprocessed_img = load_and_preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(preprocessed_img)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Find the last convolutional layer if not specified
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                print(f"Using last convolutional layer: {layer_name}")
                break
    
    # Get the gradient of the predicted class with respect to the output feature map of the last conv layer
    with tf.GradientTape() as tape:
        # Create a model that outputs both the last conv layer and the final output
        last_conv_layer = model.get_layer(layer_name)
        last_conv_output = last_conv_layer.output
        
        # Create a model that goes from the input to the last conv layer output
        last_conv_model = tf.keras.models.Model(model.inputs, last_conv_output)
        
        # Get the last conv layer output for the input image
        last_conv_output_value = last_conv_model(preprocessed_img)
        
        # Create a model that goes from the input to the final output
        final_model = tf.keras.models.Model(model.inputs, model.output)
        
        # Set the last conv layer output as a watched variable
        tape.watch(last_conv_output_value)
        
        # Get the final output for the input image
        preds = final_model(preprocessed_img)
        
        # Get the predicted class output
        pred_index = predicted_class
        class_channel = preds[:, pred_index]
    
    # Gradient of the predicted class with respect to the last conv layer output
    grads = tape.gradient(class_channel, last_conv_output_value)
    
    # Global average pooling of the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by the computed gradient values
    last_conv_output_value = last_conv_output_value[0]
    heatmap = last_conv_output_value @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap_rgb = np.uint8(255 * heatmap_resized)
    heatmap_rgb = cv2.applyColorMap(heatmap_rgb, cv2.COLORMAP_JET)
    
    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(
        cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR), 0.6, 
        heatmap_rgb, 0.4, 0
    )
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    # Create visualization with prediction info
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB))
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')
    
    # Superimposed
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title(f'Prediction: {CLASS_NAMES[predicted_class]}\nConfidence: {confidence:.2f}')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save or show the visualization
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    return superimposed_img

def process_multiple_images(image_paths, model_path, output_dir, layer_name='conv2d_2'):
    """
    Process multiple images and generate Grad-CAM visualizations.
    
    Args:
        image_paths (list): List of paths to input images.
        model_path (str): Path to the saved model.
        output_dir (str): Directory to save visualizations.
        layer_name (str): Name of the last convolutional layer.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    for i, img_path in enumerate(image_paths):
        # Generate output path
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"gradcam_{img_name}")
        
        # Apply Grad-CAM
        try:
            apply_gradcam(img_path, model_path, output_path, layer_name)
            print(f"Processed image {i+1}/{len(image_paths)}: {img_name}")
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")

def main():
    """Main function to process a set of sample images."""
    # Create output directory if it doesn't exist
    output_dir = "models/cnn_model/results/figures/grad_cam"
    os.makedirs(output_dir, exist_ok=True)
    
    # Example images from each class
    image_paths = [
        os.path.join(DATA_DIR, "Original/Benign/WBC-Benign-001.jpg"),
        os.path.join(DATA_DIR, "Original/Early/WBC-Malignant-Early-001.jpg"),
        os.path.join(DATA_DIR, "Original/Pre/WBC-Malignant-Pre-001.jpg"),
        os.path.join(DATA_DIR, "Original/Pro/WBC-Malignant-Pro-001.jpg")
    ]
    
    # Process the images
    process_multiple_images(image_paths, MODEL_PATH, output_dir, layer_name='conv2d_2')
    
    print(f"Grad-CAM visualizations saved to {output_dir}")
    return 0

if __name__ == "__main__":
    main() 