import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model

from models.efficientnet_model.scripts.preprocess import load_and_preprocess_data
from models.efficientnet_model.config import MODEL_PATH, CLASS_NAMES, IMAGE_SIZE, LEARNING_RATE, DATA_DIR
from tensorflow.keras.optimizers import Adam

def generate_confusion_matrix_figure(y_true, y_pred, class_names, filename):
    """Generates and saves a confusion matrix plot."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(filename)
    plt.close()

def generate_performance_table(metrics, filename):
    """Generates and saves a performance metrics table."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    df = pd.DataFrame(metrics, index=['Metrics'])
    df.to_csv(filename)

def evaluate_model(model_path, X_test, y_test, class_names):
    """Evaluates the trained CNN model."""
    # Load the model
    model = load_model(model_path)
    # ADDED
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Make predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Classification Report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Save report in a tabular format
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join("models", "efficientnet_model", "results", "tables", "all_performance.csv")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report_df.to_csv(report_path)
    print(f"Classification report saved to {report_path}")

    # Confusion Matrix
    confusion_matrix_filename = os.path.join("models", "efficientnet_model", "results", "figures", "confusion_matrix_ALL.png")
    generate_confusion_matrix_figure(y_true, y_pred, class_names, confusion_matrix_filename)
    print(f"Confusion matrix saved to {confusion_matrix_filename}")

    # Performance Table
    performance_metrics = {
        'accuracy': accuracy,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score']
    }
    performance_table_filename = os.path.join("models", "efficientnet_model", "results", "tables", "all_performance.csv")
    generate_performance_table(performance_metrics, performance_table_filename)
    print(f"Performance metrics saved to {performance_table_filename}")
    
def main():
    """Main function to be called from other modules."""
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return 1
        
    # 1. Load Test Data
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(os.path.join(DATA_DIR, "Original"))
    except Exception as e:
        print(f"Error loading test data: {e}")
        return 1

    # 2. Evaluate
    evaluate_model(MODEL_PATH, X_test, y_test, CLASS_NAMES)

    print("Evaluation complete.")
    return 0

if __name__ == "__main__":
    main()




#OUTPUT

# PS D:\VIT_class\4_semester\AI\Project\FINAL_MODEL\model_2_EfficientNetV2_s_cnn\Leukemia_Detection_CNN> python -m scripts.evaluate        
# 2025-03-09 16:01:07.471765: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-03-09 16:01:14.468452: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# Found classes: ['Benign', 'Early', 'Pre', 'Pro']
# Loading images from: data/ALL_dataset\Original\Benign
# Loading Benign: 100%|████████████████████████████████████████████████████████████████████████████████| 504/504 [00:00<00:00, 677.68it/s]
# Loading images from: data/ALL_dataset\Original\Early
# Loading Early: 100%|█████████████████████████████████████████████████████████████████████████████████| 985/985 [00:01<00:00, 785.55it/s]
# Loading images from: data/ALL_dataset\Original\Pre
# Loading Pre: 100%|███████████████████████████████████████████████████████████████████████████████████| 963/963 [00:01<00:00, 641.34it/s]
# Loading images from: data/ALL_dataset\Original\Pro
# Loading Pro: 100%|███████████████████████████████████████████████████████████████████████████████████| 804/804 [00:01<00:00, 764.26it/s]
# Training data shape: (2083, 224, 224, 3), (2083, 4)
# Validation data shape: (521, 224, 224, 3), (521, 4)
# Testing data shape: (652, 224, 224, 3), (652, 4)
# 2025-03-09 16:01:37.154475: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
# Test Loss: 0.0369
# Test Accuracy: 0.9862
# 21/21 ━━━━━━━━━━━━━━━━━━━━ 20s 812ms/step

# Classification Report:
#               precision    recall  f1-score   support

#       Benign       1.00      0.99      1.00       101
#        Early       0.97      1.00      0.98       197
#          Pre       1.00      0.96      0.98       193
#          Pro       0.99      1.00      0.99       161

#     accuracy                           0.99       652
#    macro avg       0.99      0.99      0.99       652
# weighted avg       0.99      0.99      0.99       652

# Classification report saved to results/tables/all_performance.csv
# Confusion matrix saved to results\figures\confusion_matrix_ALL.png
# Performance metrics saved to results\tables\all_performance.csv
# Evaluation complete.