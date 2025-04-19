# Leukemia Detection Project: Comprehensive Summary

## Project Overview
This project implements and compares two deep learning models for automated detection and classification of leukemia cells from microscopic images:

1. **Enhanced EfficientNetV2-S Model**: A transfer learning approach using a pre-trained model with custom classification layers.
2. **Traditional CNN Model**: A lightweight convolutional neural network built from scratch.

## Dataset
The project uses the ALL (Acute Lymphoblastic Leukemia) dataset which contains 3,256 cell images across four classes:
- Benign: 504 images (normal lymphocytes)
- Early: 985 images (early-stage leukemia cells)
- Pre: 963 images (pre-cancerous leukemia cells)
- Pro: 804 images (progressive/proliferative leukemia cells)

## Key Findings

### Model Performance
- The Enhanced EfficientNetV2-S model achieved **98.93%** accuracy
- The Traditional CNN model achieved **89.42%** accuracy
- Both models outperformed several baseline approaches from the literature

### Model Interpretability
- Grad-CAM visualizations revealed that:
  - EfficientNetV2-S model focuses more precisely on nuclear features
  - Traditional CNN model has more diffuse attention patterns
  - The models' focus areas correlate with pathologically significant regions

## Documentation

### Figures
All project figures are available in the `docs/assets/figures` directory:
1. Project Workflow Diagram
2. Dataset Description and Preprocessing Steps
3. Leukemia Types
4. Enhanced EfficientNetV2-S Architecture
5. Training and validation accuracy curves for EfficientNetV2-S
6. Training and validation loss curves for EfficientNetV2-S
7. Training and validation accuracy curves for Traditional CNN
8. Training and validation loss curves for Traditional CNN
9. Confusion matrix for EfficientNetV2-S model
10. Confusion matrix for Traditional CNN model
11. Performance comparison between both models
12. Comparative analysis of model performance metrics
13. Radar chart comparing accuracy metrics across models
14. Grad-CAM visualizations for EfficientNetV2-S
15. Grad-CAM visualizations for Traditional CNN

### Tables
All performance metrics and model details are available in the `docs/assets/tables` directory:
1. Summary of Previous Approaches in Leukemia Detection
2. Image Preprocessing Steps for the ALL Dataset
3. Enhanced EfficientNetV2-S Model Architecture Details
4. Traditional CNN Model Architecture Details
5. Dataset Summary for ALL Classification
6. Hyperparameter Configuration for Model Training
7. Evaluation Metrics Used in the Study
8. Classification Performance by Class for Enhanced EfficientNetV2-S
9. Classification Performance by Class for Traditional CNN
10. Comparative Performance Analysis with Existing Approaches
11. Comparison of Model Complexity and Inference Speed
12. Comparison of different models on leukemia detection datasets

## Conclusion
The project demonstrates that:
1. Deep learning models can effectively automate leukemia cell classification
2. Transfer learning with EfficientNetV2-S significantly outperforms traditional CNN approaches
3. The enhanced approach provides both superior accuracy and better interpretability
4. Model attention aligns with medically significant features in leukemia diagnosis

The results suggest strong potential for clinical application in supporting pathologists and improving diagnostic accuracy for leukemia subtypes. 