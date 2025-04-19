import os
import argparse
import subprocess
import glob

def run_batch_gradcam(model_path, image_dir, output_dir, image_pattern="*.jpg", limit=None):
    """
    Run Grad-CAM visualization for multiple images.
    
    Args:
        model_path (str): Path to the model file.
        image_dir (str): Directory containing the images.
        output_dir (str): Directory to save the visualizations.
        image_pattern (str): Pattern to match image files.
        limit (int, optional): Maximum number of images to process.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images matching the pattern
    image_paths = glob.glob(os.path.join(image_dir, "**", image_pattern), recursive=True)
    
    # Limit the number of images if specified
    if limit is not None and limit > 0:
        image_paths = image_paths[:limit]
    
    print(f"Found {len(image_paths)} images to process")
    
    # Process each image
    for i, image_path in enumerate(image_paths):
        # Generate output path
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"gradcam_{image_name}")
        
        # Run the Grad-CAM script
        cmd = ["python", "direct_gradcam.py", 
               "--model", model_path, 
               "--image", image_path, 
               "--output", output_path]
        
        print(f"Processing image {i+1}/{len(image_paths)}: {image_name}")
        subprocess.run(cmd)
        print(f"Saved visualization to {output_path}")
        print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Run Grad-CAM visualization for multiple images")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the visualizations")
    parser.add_argument("--pattern", type=str, default="*.jpg", help="Pattern to match image files")
    parser.add_argument("--limit", type=int, help="Maximum number of images to process")
    
    args = parser.parse_args()
    
    # Run batch Grad-CAM
    run_batch_gradcam(args.model, args.image_dir, args.output_dir, args.pattern, args.limit)

if __name__ == "__main__":
    main() 