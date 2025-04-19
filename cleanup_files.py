"""
Cleanup Script for Leukemia Detection Project

This script removes unwanted files and directories from the project to maintain a clean structure.
"""

import os
import shutil
import glob

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def remove_file(file_path):
    """Remove a file if it exists."""
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Removed file: {file_path}")
            return True
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")
    return False

def remove_directory(dir_path):
    """Remove a directory and all its contents if it exists."""
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"Removed directory: {dir_path}")
            return True
        except Exception as e:
            print(f"Error removing directory {dir_path}: {e}")
    return False

def main():
    """Main function to perform cleanup."""
    print_header("CLEANING UP PROJECT DIRECTORY")
    
    # Files to delete
    files_to_delete = [
        # Project structure and directory inspection files
        "../model_1_cnn/know_existing_directory_structure.txt",
        "../model_1_cnn/know_existing_directory_structure.py",
        "../model_1_cnn/create_project_structure.py",
        "../model_2_EfficientNetV2_s_cnn/know_contents_of_directory.py",
        "../model_2_EfficientNetV2_s_cnn/know_contents_of_directory.txt",
        "../model_2_EfficientNetV2_s_cnn/know_directory_structure.txt",
        "../model_2_EfficientNetV2_s_cnn/create_project_structure.py",
        "../model_2_EfficientNetV2_s_cnn/know_existing_directory_structure.py",
        "../create_directories.ps1",
        "../copy_files.ps1",
        "../prompt",
        "../recommended_directory_structure.md",
        "../new_directory_structure.md",
        "../structure.md",
        "../create_structure.py",
        "../setup_tables.py",
        "../setup_figures.py",
        "../cleanup.py",
        
        # Redundant GradCAM files
        "../run_batch_gradcam.py",
        "../simple_gradcam.py",
        "../README_gradcam.md",
        
        # XML and HTML files
        "../fig4.xml",
        "../fig5.xml",
        "../index.html",
        "../fig15-16"
    ]
    
    # Process each file
    files_removed = 0
    for file_path in files_to_delete:
        if remove_file(file_path):
            files_removed += 1
    
    # Directories to delete
    dirs_to_delete = []  # Add any directories that should be completely removed
    
    # Process each directory
    dirs_removed = 0
    for dir_path in dirs_to_delete:
        if remove_directory(dir_path):
            dirs_removed += 1
    
    print_header("CLEANUP SUMMARY")
    print(f"Removed {files_removed} files and {dirs_removed} directories.")
    print("The project structure is now clean and organized.")
    
    return 0

if __name__ == "__main__":
    main() 