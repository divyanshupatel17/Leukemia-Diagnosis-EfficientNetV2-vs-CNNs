"""
Interactive Run Guide for Leukemia Detection Project

This script provides an interactive menu to run the leukemia detection project 
in the correct sequence, allowing direct execution of commands.
"""

import os
import sys
import subprocess
import time

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def print_step(number, title, description):
    """Print a formatted step with number."""
    print(f"[{number}] {title}")
    print(f"    - {description}")

def execute_command(cmd):
    """Execute a command and display its output."""
    print_header(f"Executing: {cmd}")
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Stream output in real-time
    for line in process.stdout:
        print(line, end='')
    
    # Wait for process to complete
    process.wait()
    
    if process.returncode == 0:
        print("\nCommand completed successfully.")
    else:
        print(f"\nCommand failed with return code {process.returncode}")
    
    return process.returncode

def print_menu():
    """Print the main menu."""
    print_header("LEUKEMIA DETECTION PROJECT - INTERACTIVE RUN GUIDE")
    
    print("This guide allows you to run the leukemia detection project in the correct sequence.\n")
    print("Select a step number to execute it, or 0 to exit.\n")
    
    print_header("SETUP")
    print_step(1, "Create directory structure", 
              "Cleans up unnecessary directories and creates the proper directory structure")
    
    print_header("CNN MODEL")
    print_step(2, "Preprocess for CNN Model", 
              "Preprocesses the dataset for the Traditional CNN model")
    print_step(3, "Train CNN Model", 
              "Trains the Traditional CNN model for 20 epochs")
    print_step(4, "Evaluate CNN Model", 
              "Evaluates the Traditional CNN model and generates metrics")
    print_step(5, "Generate predictions with CNN Model", 
              "Makes predictions using the Traditional CNN model")
    print_step(6, "Generate GradCAM visualization for CNN Model", 
              "Creates visual explanation using Grad-CAM for CNN model (Figure 16)")
    
    print_header("EFFICIENTNET MODEL")
    print_step(7, "Preprocess for EfficientNet Model", 
              "Preprocesses the dataset for the Enhanced EfficientNetV2-S model")
    print_step(8, "Train EfficientNet Model", 
              "Trains the Enhanced EfficientNetV2-S model for 20 epochs")
    print_step(9, "Evaluate EfficientNet Model", 
              "Evaluates the Enhanced EfficientNetV2-S model and generates metrics")
    print_step(10, "Generate predictions with EfficientNet Model", 
               "Makes predictions using the Enhanced EfficientNetV2-S model")
    print_step(11, "Generate GradCAM visualization for EfficientNet Model", 
               "Creates visual explanation using Grad-CAM for EfficientNet model (Figure 15)")
    
    print_header("COMPARE RESULTS")
    print_step(12, "Compare model performances", 
               "Generates comparison figures between CNN and EfficientNet models (Figure 12)")
    print_step(13, "Generate model architecture diagrams", 
               "Creates detailed architecture visualizations (Figures 4 and 5)")
    print_step(14, "Generate training curves", 
               "Creates accuracy and loss curves for both models (Figures 6-9)")
    print_step(15, "Generate confusion matrices", 
               "Creates confusion matrices for both models (Figures 10-11)")
    print_step(16, "Generate comparative analysis", 
               "Creates comparative analysis across all evaluated models (Figure 13)")
    print_step(17, "Generate radar chart comparison", 
               "Creates radar chart comparing metrics across models (Figure 14)")
    print_step(18, "Generate combined GradCAM visualizations", 
               "Creates side-by-side GradCAM visualizations for both models")
    
    print_header("PROJECT DOCUMENTATION")
    print_step(19, "Generate project workflow diagram", 
               "Creates the project workflow visualization (Figure 1)")
    print_step(20, "Generate dataset description", 
               "Creates dataset description and preprocessing steps visualization (Figure 2)")
    print_step(21, "Generate leukemia types visualization", 
               "Creates visualization of different leukemia types (Figure 3)")
    print_step(22, "Generate tables", 
               "Creates all 12 tables with model performance and comparison data")
    
    print_header("ALL-IN-ONE")
    print_step(23, "Run all CNN model steps (2-6)", 
               "Runs preprocessing, training, evaluation, prediction, and GradCAM for CNN")
    print_step(24, "Run all EfficientNet model steps (7-11)", 
               "Runs preprocessing, training, evaluation, prediction, and GradCAM for EfficientNet")
    print_step(25, "Run all visualization steps (12-22)", 
               "Generates all figures and tables")
    print_step(26, "Run entire pipeline (1-22)", 
               "Runs all steps from directory setup to visualization generation")
    
    print_header("COMMANDS")
    print_step(0, "Exit", "Exit the interactive guide")

def run_step(step):
    """Run the selected step."""
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Create proper path to main.py
    main_script = os.path.join(current_dir, "main.py")
    
    if step == 1:
        execute_command(f'python "{main_script}" --operation cleanup')
    
    # CNN Model Steps
    elif step == 2:
        execute_command(f'python "{main_script}" --model cnn_model --operation preprocess')
    elif step == 3:
        execute_command(f'python "{main_script}" --model cnn_model --operation train')
    elif step == 4:
        execute_command(f'python "{main_script}" --model cnn_model --operation evaluate')
    elif step == 5:
        execute_command(f'python "{main_script}" --model cnn_model --operation predict')
    elif step == 6:
        execute_command(f'python "{main_script}" --model cnn_model --operation gradcam')
    
    # EfficientNet Model Steps
    elif step == 7:
        execute_command(f'python "{main_script}" --model efficientnet_model --operation preprocess')
    elif step == 8:
        execute_command(f'python "{main_script}" --model efficientnet_model --operation train')
    elif step == 9:
        execute_command(f'python "{main_script}" --model efficientnet_model --operation evaluate')
    elif step == 10:
        execute_command(f'python "{main_script}" --model efficientnet_model --operation predict')
    elif step == 11:
        execute_command(f'python "{main_script}" --model efficientnet_model --operation gradcam')
    
    # Visualization and Comparison Steps
    elif step == 12:
        execute_command(f'python "{main_script}" --operation compare_performance')
    elif step == 13:
        visualization_script = os.path.join(current_dir, "visualization", "model_diagrams", "create_model_architecture_diagrams.py")
        execute_command(f'python "{visualization_script}"')
    elif step == 14:
        execute_command(f'python "{main_script}" --operation generate_training_curves')
    elif step == 15:
        execute_command(f'python "{main_script}" --operation generate_confusion_matrices')
    elif step == 16:
        execute_command(f'python "{main_script}" --operation comparative_analysis')
    elif step == 17:
        execute_command(f'python "{main_script}" --operation generate_radar_chart')
    elif step == 18:
        execute_command(f'python "{main_script}" --operation combined_gradcam')
    
    # Project Documentation Steps
    elif step == 19:
        execute_command(f'python "{main_script}" --operation generate_workflow_diagram')
    elif step == 20:
        execute_command(f'python "{main_script}" --operation generate_dataset_description')
    elif step == 21:
        execute_command(f'python "{main_script}" --operation generate_leukemia_types')
    elif step == 22:
        execute_command(f'python "{main_script}" --operation generate_tables')
    
    # All-in-one Steps
    elif step == 23:
        for i in range(2, 7):
            run_step(i)
    elif step == 24:
        for i in range(7, 12):
            run_step(i)
    elif step == 25:
        for i in range(12, 23):
            run_step(i)
    elif step == 26:
        for i in range(1, 23):
            run_step(i)
    else:
        print("Invalid step number.")

def display_tables_info():
    """Display information about all tables in the project."""
    tables = [
        "Table I: Summary of Previous Approaches in Leukemia Detection",
        "Table II: Image Preprocessing Steps for the ALL Dataset",
        "Table III: Enhanced EfficientNetV2-S Model Architecture Details",
        "Table IV: Traditional CNN Model Architecture Details",
        "Table V: Dataset Summary for ALL Classification",
        "Table VI: Hyperparameter Configuration for Model Training",
        "Table VII: Evaluation Metrics Used in the Study",
        "Table VIII: Classification Performance by Class for Enhanced EfficientNetV2-S",
        "Table IX: Classification Performance by Class for Traditional CNN",
        "Table X: Comparative Performance Analysis with Existing Approaches",
        "Table XI: Comparison of Model Complexity and Inference Speed",
        "Table XII: Comparison of different models on leukemia detection datasets"
    ]
    
    print_header("PROJECT TABLES")
    for i, table in enumerate(tables, 1):
        print(f"Table {i}: {table}")
    print("\n")

def display_figures_info():
    """Display information about all figures in the project."""
    figures = [
        "Project Workflow Diagram",
        "Dataset Description and Preprocessing Steps",
        "Leukemia Types",
        "Enhanced EfficientNetV2-S Architecture for Multi-Class Leukemia Classification",
        "Training and validation accuracy curves over 20 epochs for Enhanced EfficientNetV2-S",
        "Training and validation loss curves over 20 epochs for Enhanced EfficientNetV2-S",
        "Training and validation accuracy curves over 20 epochs for Traditional CNN",
        "Training and validation loss curves over 20 epochs for Traditional CNN",
        "Confusion matrix for Enhanced EfficientNetV2-S model",
        "Confusion matrix for Traditional CNN model",
        "Performance comparison between Enhanced EfficientNetV2-S and Traditional CNN",
        "Comparative analysis of model performance metrics across all evaluated models",
        "Radar chart comparing accuracy, precision, recall, and F1-score across models",
        "Grad-CAM Visualizations for Enhanced EfficientNetV2-S",
        "Grad-CAM Visualizations for Traditional CNN"
    ]
    
    print_header("PROJECT FIGURES")
    for i, figure in enumerate(figures, 1):
        print(f"Figure {i}: {figure}")
    print("\n")

def show_project_info():
    """Display general information about the project."""
    print_header("PROJECT INFORMATION")
    print("Enhanced EfficientNetV2-S Architecture for Multi-Class Leukemia Diagnosis Using the ALL Dataset")
    print("\nThis project implements and compares two models for leukemia classification:")
    print("1. Enhanced EfficientNetV2-S - A transfer learning model with 98.93% accuracy")
    print("2. Traditional CNN - A baseline model with 89.42% accuracy")
    print("\nThe models are trained on the ALL dataset with 4 classes of leukemia cells:")
    print("- Benign: Normal lymphocytes or non-cancerous cells")
    print("- Early: Early-stage leukemia cells")
    print("- Pre: Pre-cancerous leukemia cells")
    print("- Pro: Progressive (Proliferative) leukemia cells")
    print("\n")
    
    display_tables_info()
    display_figures_info()

def main():
    """Main function to run the interactive guide."""
    # Ensure we're in the leukemia_detection directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print_header("LEUKEMIA DETECTION PROJECT")
    print(f"Current working directory: {os.getcwd()}")
    print("This interactive guide will help you run the leukemia detection project components.")
    
    while True:
        print_menu()
        try:
            user_input = input("\nEnter step number (0 to exit, i for project info): ")
            if user_input.lower() == 'i':
                show_project_info()
            else:
                choice = int(user_input)
                if choice == 0:
                    print("Exiting the interactive guide. Goodbye!")
                    return 0
                run_step(choice)
            input("\nPress Enter to continue...")
        except ValueError:
            print("Please enter a valid number or 'i' for project info.")
        except KeyboardInterrupt:
            print("\nOperation interrupted by user.")
            return 1
        except Exception as e:
            print(f"An error occurred: {e}")
            return 1

if __name__ == "__main__":
    sys.exit(main()) 