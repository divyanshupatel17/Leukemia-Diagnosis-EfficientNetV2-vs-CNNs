"""
Direct GradCAM Implementation

This module implements the Gradient-weighted Class Activation Mapping (Grad-CAM) technique
for creating visual explanations for decisions made by CNN models.

Reference:
Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017).
Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.
IEEE International Conference on Computer Vision (ICCV).
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import argparse

# Define class names
CLASS_NAMES = ["Benign", "Early", "Pre", "Pro"]

def load_and_preprocess_image(img_path, target_size=(128, 128)):
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
        # Load image using PIL
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

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.
    Creates visual explanations for decisions made by CNN models.
    """
    
    def __init__(self, model, last_conv_layer_name):
        """
        Initialize the GradCAM object.
        
        Args:
            model: A tf.keras.Model instance
            last_conv_layer_name: Name of the last convolutional layer in the model
        """
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        
        # Get the model's output and the specified convolutional layer
        self.grad_model = self._create_gradient_model()
    
    def _create_gradient_model(self):
        """
        Creates a model that outputs the activations of the last convolutional layer
        and the final output predictions.
        
        Returns:
            A tf.keras.Model that computes both the last conv layer activations
            and the model's predictions
        """
        # Get the specified convolutional layer
        last_conv_layer = self.model.get_layer(self.last_conv_layer_name)
        
        # Create a model that outputs both the last conv layer output and the final predictions
        return tf.keras.Model(
            inputs=[self.model.inputs],
            outputs=[last_conv_layer.output, self.model.output]
        )
    
    def compute_heatmap(self, image, class_idx=None, eps=1e-8):
        """
        Compute the Grad-CAM heatmap for an image.
        
        Args:
            image: Input image as a numpy array, shaped (1, height, width, channels)
            class_idx: Index of the class to generate heatmap for. If None, uses the predicted class.
            eps: Small value to avoid division by zero
            
        Returns:
            The normalized heatmap as a numpy array, shaped (height, width)
        """
        # Get the model's prediction for this image
        if class_idx is None:
            prediction = self.model.predict(image)
            class_idx = np.argmax(prediction[0])
        
        # Compute gradients of the predicted class with respect to the last conv layer output
        with tf.GradientTape() as tape:
            # Cast image to float32
            if image.dtype != tf.float32:
                image = tf.cast(image, tf.float32)
            
            # Forward pass to get conv output and predictions
            conv_output, predictions = self.grad_model(image)
            
            # Get the score for the target class
            if isinstance(predictions, list):
                predictions = predictions[0]  # Take the first output if model has multiple outputs
            
            class_channel = predictions[:, class_idx]
        
        # Gradients of the target class with respect to the conv layer output
        grads = tape.gradient(class_channel, conv_output)
        
        # Compute guided gradients
        gate_f = tf.cast(conv_output > 0, tf.float32)
        gate_r = tf.cast(grads > 0, tf.float32)
        guided_grads = gate_f * gate_r * grads
        
        # Get the conv layer output as numpy array
        conv_output = conv_output[0].numpy()
        guided_grads = guided_grads[0].numpy()
        
        # Compute weights: global average pooling of gradients
        weights = np.mean(guided_grads, axis=(0, 1))
        
        # Create heatmap: weighted sum of feature maps
        cam = np.zeros(conv_output.shape[0:2], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * conv_output[:, :, i]
        
        # Apply ReLU to the heatmap
        cam = np.maximum(cam, 0)
        
        # Normalize the heatmap to [0, 1]
        if np.max(cam) > eps:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + eps)
        
        return cam

def direct_gradcam(model_path, image_path, output_path=None):
    """
    Generate a Grad-CAM visualization for a given model and image using a direct approach.
    
    Args:
        model_path (str): Path to the model file.
        image_path (str): Path to the image file.
        output_path (str, optional): Path to save the visualization.
    """
    # Load the model
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    
    # Determine if this is the EfficientNetV2 model
    is_efficientnet = "efficientnetv2" in model_path.lower()
    target_size = (224, 224) if is_efficientnet else (128, 128)
    
    # Load and preprocess the image
    print(f"Processing image: {image_path}")
    original_img, preprocessed_img = load_and_preprocess_image(image_path, target_size)
    
    # Make prediction
    print("Making prediction...")
    predictions = model.predict(preprocessed_img, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    print(f"Predicted class: {CLASS_NAMES[predicted_class]} with confidence: {confidence:.2f}")
    
    # Get the last convolutional layer name
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            last_conv_layer_name = layer.name
            break
    
    if last_conv_layer_name is None:
        print("Could not find a convolutional layer in the model.")
        return
    
    # Create GradCAM
    grad_cam = GradCAM(model, last_conv_layer_name)
    
    # Compute heatmap
    heatmap = grad_cam.compute_heatmap(preprocessed_img)
    
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
    plt.title('Feature Map Heatmap')
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

def main():
    parser = argparse.ArgumentParser(description="Generate direct feature map visualization for a model and image")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--output", type=str, help="Path to save the visualization (optional)")
    
    args = parser.parse_args()
    
    # Create output directory if specified
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Generate visualization
    direct_gradcam(args.model, args.image, args.output)

if __name__ == "__main__":
    main() 