import os
import sys
import argparse
import glob

def find_sample_images(data_dir, num_samples=3):
    """
    Find sample images from each class in the dataset.
    
    Args:
        data_dir (str): Path to the dataset directory.
        num_samples (int): Number of samples to select from each class.
        
    Returns:
        list: List of paths to sample images.
    """
    classes = ["Benign", "Early", "Pre", "Pro"]
    sample_images = []
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, "Original", class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory not found: {class_dir}")
            continue
        
        # Get all image files in the class directory
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(glob.glob(os.path.join(class_dir, ext)))
        
        # Select samples
        selected_samples = image_files[:num_samples]
        sample_images.extend(selected_samples)
        
        print(f"Selected {len(selected_samples)} samples from {class_name} class")
    
    return sample_images

def find_model_file(model_name, possible_locations):
    """
    Find a model file in possible locations.
    
    Args:
        model_name (str): Name of the model file.
        possible_locations (list): List of possible directory paths.
        
    Returns:
        str: Path to the model file if found, None otherwise.
    """
    for location in possible_locations:
        model_path = os.path.join(location, model_name)
        if os.path.exists(model_path):
            print(f"Found model at: {model_path}")
            return model_path
    
    return None

def run_traditional_cnn_gradcam(image_paths, model_path=None):
    """
    Run Grad-CAM visualization for the Traditional CNN model.
    
    Args:
        image_paths (list): List of paths to input images.
        model_path (str, optional): Path to the model file.
    """
    print("\n=== Running Grad-CAM for Traditional CNN Model ===")
    
    # Convert image paths to absolute paths
    abs_image_paths = [os.path.abspath(path) for path in image_paths]
    
    # Change to the model_1_cnn directory
    original_dir = os.getcwd()
    script_dir = os.path.join(original_dir, "model_1_cnn", "Leukemia_Detection_CNN", "scripts")
    os.chdir(script_dir)
    
    # Run the Grad-CAM script
    cmd = "python grad_cam_visualization.py"
    
    # Add model path if provided
    if model_path:
        abs_model_path = os.path.abspath(os.path.join(original_dir, model_path))
        cmd += f" --model \"{abs_model_path}\""
    
    # Add image paths
    for path in abs_image_paths:
        cmd += f" \"{path}\""
    
    print(f"Executing: {cmd}")
    os.system(cmd)
    
    # Return to the original directory
    os.chdir(original_dir)

def run_efficientnetv2s_gradcam(image_paths, model_path=None):
    """
    Run Grad-CAM visualization for the EfficientNetV2-S model.
    
    Args:
        image_paths (list): List of paths to input images.
        model_path (str, optional): Path to the model file.
    """
    print("\n=== Running Grad-CAM for EfficientNetV2-S Model ===")
    
    # Convert image paths to absolute paths
    abs_image_paths = [os.path.abspath(path) for path in image_paths]
    
    # Change to the model_2_EfficientNetV2_s_cnn directory
    original_dir = os.getcwd()
    script_dir = os.path.join(original_dir, "model_2_EfficientNetV2_s_cnn", "Leukemia_Detection_CNN", "scripts")
    os.chdir(script_dir)
    
    # Run the Grad-CAM script
    cmd = "python grad_cam_visualization.py"
    
    # Add model path if provided
    if model_path:
        abs_model_path = os.path.abspath(os.path.join(original_dir, model_path))
        cmd += f" --model \"{abs_model_path}\""
    
    # Add image paths
    for path in abs_image_paths:
        cmd += f" \"{path}\""
    
    print(f"Executing: {cmd}")
    os.system(cmd)
    
    # Return to the original directory
    os.chdir(original_dir)

def main():
    """Main function to run both Grad-CAM visualizations."""
    parser = argparse.ArgumentParser(description="Run Grad-CAM visualizations for both models")
    parser.add_argument("--data_dir", type=str, default="model_1_cnn/Leukemia_Detection_CNN/data/ALL_dataset",
                        help="Path to the dataset directory")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of samples to select from each class")
    parser.add_argument("--images", nargs="+", default=None,
                        help="Specific image paths to use (optional)")
    parser.add_argument("--cnn_model", type=str, default=None,
                        help="Path to the Traditional CNN model file (optional)")
    parser.add_argument("--efficientnet_model", type=str, default=None,
                        help="Path to the EfficientNetV2-S model file (optional)")
    parser.add_argument("--skip_cnn", action="store_true",
                        help="Skip running the Traditional CNN model")
    parser.add_argument("--skip_efficientnet", action="store_true",
                        help="Skip running the EfficientNetV2-S model")
    
    args = parser.parse_args()
    
    # Try to find model files if not provided
    if not args.cnn_model:
        possible_locations = [
            "model_1_cnn/Leukemia_Detection_CNN/models/saved_models",
            "models/saved_models",
            "models",
            "."
        ]
        args.cnn_model = find_model_file("leukemia_model.h5", possible_locations)
    
    if not args.efficientnet_model:
        possible_locations = [
            "model_2_EfficientNetV2_s_cnn/Leukemia_Detection_CNN/models",
            "models",
            "."
        ]
        args.efficientnet_model = find_model_file("efficientnetv2s_leukemia_model.h5", possible_locations)
    
    # Get sample images
    if args.images:
        sample_images = args.images
        print(f"Using {len(sample_images)} provided image paths")
    else:
        sample_images = find_sample_images(args.data_dir, args.num_samples)
        print(f"Found {len(sample_images)} sample images")
    
    if not sample_images:
        print("No sample images found. Please check the dataset directory or provide specific image paths.")
        sys.exit(1)
    
    # Run Grad-CAM for both models
    if not args.skip_cnn:
        run_traditional_cnn_gradcam(sample_images, args.cnn_model)
    else:
        print("Skipping Traditional CNN model as requested")
    
    if not args.skip_efficientnet:
        run_efficientnetv2s_gradcam(sample_images, args.efficientnet_model)
    else:
        print("Skipping EfficientNetV2-S model as requested")
    
    print("\n=== Grad-CAM Visualizations Complete ===")
    print("Traditional CNN results: model_1_cnn/Leukemia_Detection_CNN/results/grad_cam_visualizations/traditional_cnn")
    print("EfficientNetV2-S results: model_2_EfficientNetV2_s_cnn/Leukemia_Detection_CNN/results/grad_cam_visualizations/efficientnetv2s")

if __name__ == "__main__":
    main() 