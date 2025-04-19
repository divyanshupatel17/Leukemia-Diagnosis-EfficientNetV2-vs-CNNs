import os
import glob
import argparse
import random
from PIL import Image

def is_valid_image(file_path):
    """
    Check if a file is a valid image.
    
    Args:
        file_path (str): Path to the file.
        
    Returns:
        bool: True if the file is a valid image, False otherwise.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except:
        return False

def find_images(directory, extensions=None, max_images=10, recursive=True):
    """
    Find image files in a directory.
    
    Args:
        directory (str): Directory to search in.
        extensions (list, optional): List of file extensions to look for.
        max_images (int, optional): Maximum number of images to return.
        recursive (bool, optional): Whether to search recursively.
        
    Returns:
        list: List of paths to image files.
    """
    if extensions is None:
        extensions = ["jpg", "jpeg", "png", "bmp", "gif"]
    
    image_files = []
    
    if recursive:
        for ext in extensions:
            pattern = os.path.join(directory, "**", f"*.{ext}")
            image_files.extend(glob.glob(pattern, recursive=True))
    else:
        for ext in extensions:
            pattern = os.path.join(directory, f"*.{ext}")
            image_files.extend(glob.glob(pattern))
    
    # Filter out invalid images
    valid_images = [f for f in image_files if is_valid_image(f)]
    
    # Limit the number of images
    if max_images > 0 and len(valid_images) > max_images:
        valid_images = random.sample(valid_images, max_images)
    
    return valid_images

def main():
    parser = argparse.ArgumentParser(description="Find sample images in the workspace")
    parser.add_argument("--directory", type=str, default=".",
                        help="Directory to search in (default: current directory)")
    parser.add_argument("--extensions", type=str, nargs="+", default=["jpg", "jpeg", "png"],
                        help="File extensions to look for (default: jpg, jpeg, png)")
    parser.add_argument("--max_images", type=int, default=10,
                        help="Maximum number of images to return (default: 10)")
    parser.add_argument("--no-recursive", action="store_true",
                        help="Don't search recursively")
    
    args = parser.parse_args()
    
    # Find images
    images = find_images(
        args.directory,
        args.extensions,
        args.max_images,
        not args.no_recursive
    )
    
    # Print results
    if images:
        print(f"Found {len(images)} image(s):")
        for img in images:
            print(f"  {img}")
        
        print("\nYou can use these images with the Grad-CAM scripts:")
        print(f"python run_gradcam_visualization.py --images {' '.join(images)}")
    else:
        print("No images found.")

if __name__ == "__main__":
    main() 