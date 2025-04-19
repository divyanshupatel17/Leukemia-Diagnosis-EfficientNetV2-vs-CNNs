import os
import sys
import argparse
import shutil

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def cleanup():
    """Clean up unnecessary files and organize the project structure."""
    print_header("Cleaning up project directory")
    
    # Files to delete (relative to project root)
    files_to_delete = [
        "model_1_cnn/know_existing_directory_structure.txt",
        "model_1_cnn/know_existing_directory_structure.py",
        "model_1_cnn/create_project_structure.py",
        "model_2_EfficientNetV2_s_cnn/know_contents_of_directory.py",
        "model_2_EfficientNetV2_s_cnn/know_contents_of_directory.txt",
        "model_2_EfficientNetV2_s_cnn/know_directory_structure.txt",
        "model_2_EfficientNetV2_s_cnn/create_project_structure.py",
        "model_2_EfficientNetV2_s_cnn/know_existing_directory_structure.py"
    ]
    
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    
    # Create directory structure
    directories = [
        # Original dataset
        "data/ALL_dataset/Original/Benign",
        "data/ALL_dataset/Original/Early",
        "data/ALL_dataset/Original/Pre",
        "data/ALL_dataset/Original/Pro",
        
        # CNN model directories
        "data/cnn_model/processed/train",
        "data/cnn_model/processed/test",
        "data/cnn_model/augmented/train",
        "data/cnn_model/train",
        "data/cnn_model/validation",
        "data/cnn_model/test",
        
        # EfficientNet model directories
        "data/efficientnet_model/processed/train",
        "data/efficientnet_model/processed/test",
        "data/efficientnet_model/augmented/train",
        "data/efficientnet_model/train",
        "data/efficientnet_model/validation",
        "data/efficientnet_model/test",
        
        # Model save directories
        "models/saved_models",
        
        # Results directories for both models
        "models/cnn_model/results/figures",
        "models/cnn_model/results/training_plots",
        "models/efficientnet_model/results/figures",
        "models/efficientnet_model/results/training_plots"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            except Exception as e:
                print(f"Error creating directory {directory}: {e}")
    
    print("Cleanup and directory setup completed successfully!")
    return 0

def run_cnn_preprocess():
    """Run preprocessing for CNN model."""
    print_header("Running preprocessing for CNN model")
    try:
        # Import the preprocess module from CNN model and run it
        from models.cnn_model.scripts import preprocess
        preprocess.main()
        return 0
    except Exception as e:
        print(f"Error preprocessing CNN model: {e}")
        return 1

def run_cnn_train():
    """Run training for CNN model."""
    print_header("Running training for CNN model")
    try:
        # Import the train module from CNN model and run it
        from models.cnn_model.scripts import train
        train.main()
        return 0
    except Exception as e:
        print(f"Error training CNN model: {e}")
        return 1

def run_cnn_evaluate():
    """Run evaluation for CNN model."""
    print_header("Running evaluation for CNN model")
    try:
        # Import the evaluate module from CNN model and run it
        from models.cnn_model.scripts import evaluate
        evaluate.main()
        return 0
    except Exception as e:
        print(f"Error evaluating CNN model: {e}")
        return 1

def run_cnn_predict():
    """Run prediction for CNN model."""
    print_header("Running prediction for CNN model")
    try:
        # Import the predict module from CNN model and run it
        from models.cnn_model.scripts import predict
        predict.main()
        return 0
    except Exception as e:
        print(f"Error making predictions with CNN model: {e}")
        return 1

def run_cnn_gradcam():
    """Run Grad-CAM visualization for CNN model."""
    print_header("Running Grad-CAM visualization for CNN model")
    try:
        # Import the gradcam module from CNN model and run it
        from models.cnn_model.scripts import grad_cam_visualization
        grad_cam_visualization.main()
        return 0
    except Exception as e:
        print(f"Error generating Grad-CAM for CNN model: {e}")
        return 1

def run_efficientnet_preprocess():
    """Run preprocessing for EfficientNet model."""
    print_header("Running preprocessing for EfficientNet model")
    try:
        # Import the preprocess module from EfficientNet model and run it
        from models.efficientnet_model.scripts import preprocess
        preprocess.main()
        return 0
    except Exception as e:
        print(f"Error preprocessing EfficientNet model: {e}")
        return 1

def run_efficientnet_train():
    """Run training for EfficientNet model."""
    print_header("Running training for EfficientNet model")
    try:
        # Import the train module from EfficientNet model and run it
        from models.efficientnet_model.scripts import train
        train.main()
        return 0
    except Exception as e:
        print(f"Error training EfficientNet model: {e}")
        return 1

def run_efficientnet_evaluate():
    """Run evaluation for EfficientNet model."""
    print_header("Running evaluation for EfficientNet model")
    try:
        # Import the evaluate module from EfficientNet model and run it
        from models.efficientnet_model.scripts import evaluate
        evaluate.main()
        return 0
    except Exception as e:
        print(f"Error evaluating EfficientNet model: {e}")
        return 1

def run_efficientnet_predict():
    """Run prediction for EfficientNet model."""
    print_header("Running prediction for EfficientNet model")
    try:
        # Import the predict module from EfficientNet model and run it
        from models.efficientnet_model.scripts import predict
        predict.main()
        return 0
    except Exception as e:
        print(f"Error making predictions with EfficientNet model: {e}")
        return 1

def run_efficientnet_gradcam():
    """Run Grad-CAM visualization for EfficientNet model."""
    print_header("Running Grad-CAM visualization for EfficientNet model")
    try:
        # Import the gradcam module from EfficientNet model and run it
        from models.efficientnet_model.scripts import grad_cam_visualization
        grad_cam_visualization.main()
        return 0
    except Exception as e:
        print(f"Error generating Grad-CAM for EfficientNet model: {e}")
        return 1

def run_compare_models():
    """Run comparison between models."""
    print_header("Running model comparison")
    try:
        from utils import compare_models
        compare_models.main()
        return 0
    except Exception as e:
        print(f"Error comparing models: {e}")
        return 1

def run_combined_gradcam():
    """Run combined Grad-CAM visualization for both models."""
    print_header("Running combined Grad-CAM visualization")
    try:
        from visualization.gradcam_figures import create_gradcam_figures
        create_gradcam_figures.main()
        return 0
    except Exception as e:
        print(f"Error generating combined Grad-CAM: {e}")
        return 1

def main():
    """Main function to parse arguments and run selected operation."""
    parser = argparse.ArgumentParser(description="Leukemia Detection project runner")
    parser.add_argument("--model", choices=["cnn_model", "efficientnet_model"], 
                        help="The model to use for the operation")
    parser.add_argument("--operation", choices=["preprocess", "train", "evaluate", "predict", 
                                              "gradcam", "cleanup", "compare", "combined_gradcam"], 
                        required=True, help="The operation to perform")
    
    args = parser.parse_args()
    
    # Operations that don't require a model argument
    if args.operation == "cleanup":
        return cleanup()
    elif args.operation == "compare":
        return run_compare_models()
    elif args.operation == "combined_gradcam":
        return run_combined_gradcam()
    
    # Operations that require a model argument
    if not args.model:
        print("Error: --model argument is required for this operation.")
        return 1
    
    # CNN model operations
    if args.model == "cnn_model":
        if args.operation == "preprocess":
            return run_cnn_preprocess()
        elif args.operation == "train":
            return run_cnn_train()
        elif args.operation == "evaluate":
            return run_cnn_evaluate()
        elif args.operation == "predict":
            return run_cnn_predict()
        elif args.operation == "gradcam":
            return run_cnn_gradcam()
    
    # EfficientNet model operations
    elif args.model == "efficientnet_model":
        if args.operation == "preprocess":
            return run_efficientnet_preprocess()
        elif args.operation == "train":
            return run_efficientnet_train()
        elif args.operation == "evaluate":
            return run_efficientnet_evaluate()
        elif args.operation == "predict":
            return run_efficientnet_predict()
        elif args.operation == "gradcam":
            return run_efficientnet_gradcam()
    
    print(f"Invalid combination of model and operation: {args.model}, {args.operation}")
    return 1

if __name__ == "__main__":
    sys.exit(main()) 