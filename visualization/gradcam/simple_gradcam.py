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

def simple_gradcam(model_path, image_path, output_path=None):
    """
    Generate a Grad-CAM visualization for a given model and image.
    
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
    
    # Find the last convolutional layer
    last_conv_layer = None
    
    if is_efficientnet:
        # For EfficientNetV2, find the base model first
        base_model = None
        for layer in model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 0:
                base_model = layer
                print(f"Found base model: {base_model.name}")
                break
        
        if base_model is not None:
            # Search in the base model
            for layer in reversed(base_model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer
                    break
    else:
        # For traditional CNN, search directly
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
    
    if last_conv_layer is None:
        print("Could not find a convolutional layer. Using the last layer before flatten.")
        for i, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.layers.Flatten):
                last_conv_layer = model.layers[i-1]
                break
    
    if last_conv_layer is None:
        raise ValueError("Could not find a suitable layer for Grad-CAM")
    
    print(f"Using layer for Grad-CAM: {last_conv_layer.name}")
    
    # Create a simplified model that outputs the activations of the last conv layer
    if is_efficientnet and base_model is not None:
        # For EfficientNetV2
        last_conv_output = base_model.get_layer(last_conv_layer.name).output
        grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[last_conv_output, model.output])
    else:
        # For traditional CNN
        last_conv_output = model.get_layer(last_conv_layer.name).output
        grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[last_conv_output, model.output])
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(preprocessed_img)
        loss = predictions[:, predicted_class]
    
    # Extract gradients
    grads = tape.gradient(loss, conv_output)
    
    # Global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by the computed gradient values
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
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

def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualization for a model and image")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--output", type=str, help="Path to save the visualization (optional)")
    
    args = parser.parse_args()
    
    # Create output directory if specified
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Generate Grad-CAM visualization
    simple_gradcam(args.model, args.image, args.output)

if __name__ == "__main__":
    main() 