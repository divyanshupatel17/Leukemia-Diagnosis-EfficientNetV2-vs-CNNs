# scripts/evaluate.py
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import seaborn as sns
import pandas as pd
from models.cnn_model.config import MODEL_PATH, CLASS_NAMES, IMAGE_SIZE, DATA_DIR, MODEL_DATA_DIR
from models.cnn_model.scripts.preprocess import load_and_preprocess_data, save_data  # Import data loading
import itertools # For plotting confusion matrix


def load_test_data(data_dir):
    """Loads and preprocesses the test data."""
    print(f"Loading test data from: {data_dir}")  # Debugging print
    X_test, X_val, X_test2, y_test, y_val, y_test2 = load_and_preprocess_data(data_dir) # Load test data  (Modified)
    print(f"X_test shape: {X_test.shape}")  # Debugging print
    print(f"y_test shape: {y_test.shape}")  # Debugging print
    return X_test, y_test

def evaluate_model(model_path, X_test, y_test, class_names):
    """
    Evaluates the trained CNN model.

    Args:
        model_path (str): Path to the saved model.
        X_test (numpy.ndarray): Test data.
        y_test (numpy.ndarray): Test labels (one-hot encoded).
        class_names (list): List of class names.

    Returns:
        None
    """
    # Create output directories if they don't exist
    os.makedirs("models/cnn_model/results/performance_metrics", exist_ok=True)
    os.makedirs("models/cnn_model/results/confusion_matrix", exist_ok=True)
    os.makedirs("models/cnn_model/results/training_plots", exist_ok=True)
    
    # Load the model
    model = load_model(model_path)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Make predictions
    y_pred_probs = model.predict(X_test) # Probabilities for each class
    y_pred = np.argmax(y_pred_probs, axis=1) # Predicted class labels (integers)
    y_true = np.argmax(y_test, axis=1) # True class labels

    # Classification Report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True) # get output as a dictionary
    print(classification_report(y_true, y_pred, target_names=class_names)) # Print for the console

    # Save report in a tabular format
    report_df = pd.DataFrame(report).transpose()  # Convert to DataFrame
    report_df.to_csv("models/cnn_model/results/performance_metrics/classification_report.csv")  # Save to CSV
    print("Classification report saved to models/cnn_model/results/performance_metrics/classification_report.csv")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix_seaborn(cm, class_names, save_path="models/cnn_model/results/confusion_matrix/confusion_matrix_seaborn.png")
    plot_roc_curve(y_test, y_pred_probs, class_names, save_path="models/cnn_model/results/training_plots/roc_curve.png") # plot the ROC plot

    # Calculate per-class accuracy from the confusion matrix
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    accuracy_df = pd.DataFrame({'Accuracy': per_class_accuracy}, index=class_names)
    accuracy_df.to_csv("models/cnn_model/results/performance_metrics/per_class_accuracy.csv")
    print("Per-Class Accuracy saved to models/cnn_model/results/performance_metrics/per_class_accuracy.csv")


    # Save metrics to a file
    with open("models/cnn_model/results/performance_metrics/metrics.txt", "w") as f:
        f.write(f"Test Loss: {loss:.4f}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names))
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm)) # Save confusion matrix as text

    print("Evaluation complete. Results saved.")


def plot_confusion_matrix(cm, class_names, normalize=False, save_path=None, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plots the confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        title = 'Normalized Confusion Matrix' # Change the title, so that both plots have different titles
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]:.2f}" if normalize else cm[i, j],  # Format text
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix_seaborn(cm, class_names, save_path=None):
    """
    Plots the confusion matrix using seaborn for a better look.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_roc_curve(y_true, y_score, class_names, save_path=None):
    """
    Plots the ROC curve for each class.
    """
    n_classes = len(class_names)
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Add a diagonal line for random guessing
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    """Main function that can be imported and called by other modules."""
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return 1
        
    # 1. Load Test Data
    data_dir = os.path.join(MODEL_DATA_DIR, "processed/test")
    
    # Check if test data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Test data directory not found at {data_dir}")
        # Try to preprocess data first
        try:
            from models.cnn_model.scripts.preprocess import main as preprocess_main
            print("Attempting to preprocess data first...")
            preprocess_main()
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            return 1
    
    print(f"Loading test data from: {data_dir}")
    try:
        X_test, y_test = load_test_data(data_dir) # Load test data
    except Exception as e:
        print(f"Error loading test data: {e}")
        return 1
        
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    # 2. Evaluate
    evaluate_model(MODEL_PATH, X_test, y_test, CLASS_NAMES)

    print("Evaluation complete.")
    return 0

if __name__ == "__main__":
    # 1. Load Test Data
    data_dir = os.path.join(MODEL_DATA_DIR, "processed/test")
    print(f"Shape of X_test before function: {data_dir}")
    X_test, y_test = load_test_data(data_dir) # Load test data
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    # 2. Evaluate
    evaluate_model(MODEL_PATH, X_test, y_test, CLASS_NAMES)

    print("Evaluation complete.")


# TERMINAL OUTPUT:

# PS D:\VIT_class\4_semester\AI\Project\temp\model_v2\Leukemia_Detection_CNN> python -m scripts.evaluate
# 2025-03-09 10:09:10.939958: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-03-09 10:09:11.771494: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# Shape of X_test before function: data/processed/test
# Loading test data from: data/processed/test
# Testing or Validation data loaded using .npy files
# X_test shape: (652, 128, 128, 3)
# y_test shape: (652, 4)
# Shape of X_test: (652, 128, 128, 3)
# Shape of y_test: (652, 4)
# 2025-03-09 10:09:15.004901: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
# Test Loss: 0.4195
# Test Accuracy: 0.8942
# 21/21 ━━━━━━━━━━━━━━━━━━━━ 1s 35ms/step 

# Classification Report:
#               precision    recall  f1-score   support

#       Benign       0.80      0.71      0.75       101
#        Early       0.83      0.89      0.86       197
#          Pre       0.97      0.91      0.94       193
#          Pro       0.94      1.00      0.97       161

#     accuracy                           0.89       652
#    macro avg       0.89      0.88      0.88       652
# weighted avg       0.89      0.89      0.89       652

# Classification report saved to results/performance_metrics/classification_report.csv
# Per-Class Accuracy saved to results/performance_metrics/per_class_accuracy.csv
# Evaluation complete. Results saved.
# Evaluation complete.