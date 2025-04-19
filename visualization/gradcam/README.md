# Grad-CAM Visualization for Leukemia Detection Models

This repository contains two deep learning models for leukemia detection:
1. Traditional CNN model
2. EfficientNetV2-S model

## Generating Grad-CAM Visualizations

We provide a script `direct_gradcam.py` that can generate Grad-CAM visualizations for both models. This script uses a simplified approach to generate heatmaps that highlight the regions of the image that the model focuses on when making predictions.

### Usage

```bash
python direct_gradcam.py --model <path_to_model> --image <path_to_image> --output <path_to_output>
```

#### Arguments:
- `--model`: Path to the model file (.h5)
- `--image`: Path to the image file (.jpg, .png)
- `--output`: (Optional) Path to save the visualization. If not provided, the visualization will be displayed.

### Examples

#### Traditional CNN Model
```bash
python direct_gradcam.py --model "model_1_cnn/Leukemia_Detection_CNN/models/saved_models/leukemia_model.h5" --image "model_1_cnn/Leukemia_Detection_CNN/data/archive/Original/Pre/WBC-Malignant-Pre-299.jpg" --output "model_1_cnn/Leukemia_Detection_CNN/results/grad_cam_visualizations/traditional_cnn/gradcam_WBC-Malignant-Pre-299.jpg"
```

#### EfficientNetV2-S Model
```bash
python direct_gradcam.py --model "model_2_EfficientNetV2_s_cnn/Leukemia_Detection_CNN/models/efficientnetv2s_leukemia_model.h5" --image "model_2_EfficientNetV2_s_cnn/Leukemia_Detection_CNN/data/ALL_dataset/Segmented/Early/WBC-Malignant-Early-900.jpg" --output "model_2_EfficientNetV2_s_cnn/Leukemia_Detection_CNN/results/grad_cam_visualizations/efficientnetv2s/gradcam_WBC-Malignant-Early-900.jpg"
```

## Batch Processing

For processing multiple images at once, we provide a batch script `run_batch_gradcam.py` that can run the Grad-CAM visualization for multiple images in a directory.

### Usage

```bash
python run_batch_gradcam.py --model <path_to_model> --image_dir <path_to_image_directory> --output_dir <path_to_output_directory> [--pattern <image_pattern>] [--limit <max_images>]
```

#### Arguments:
- `--model`: Path to the model file (.h5)
- `--image_dir`: Directory containing the images
- `--output_dir`: Directory to save the visualizations
- `--pattern`: (Optional) Pattern to match image files (default: "*.jpg")
- `--limit`: (Optional) Maximum number of images to process

### Examples

#### Traditional CNN Model
```bash
python run_batch_gradcam.py --model "model_1_cnn/Leukemia_Detection_CNN/models/saved_models/leukemia_model.h5" --image_dir "model_1_cnn/Leukemia_Detection_CNN/data/archive/Original/Pre" --output_dir "model_1_cnn/Leukemia_Detection_CNN/results/grad_cam_visualizations/traditional_cnn" --limit 5
```

#### EfficientNetV2-S Model
```bash
python run_batch_gradcam.py --model "model_2_EfficientNetV2_s_cnn/Leukemia_Detection_CNN/models/efficientnetv2s_leukemia_model.h5" --image_dir "model_2_EfficientNetV2_s_cnn/Leukemia_Detection_CNN/data/ALL_dataset/Segmented/Early" --output_dir "model_2_EfficientNetV2_s_cnn/Leukemia_Detection_CNN/results/grad_cam_visualizations/efficientnetv2s" --limit 5
```

### Finding Sample Images

You can use the `find_sample_images.py` script to find sample images in the dataset:

```bash
python find_sample_images.py
```

This will output a list of sample images that you can use with the Grad-CAM script.

### Output

The script generates a visualization with three panels:
1. Original image
2. Grad-CAM heatmap
3. Superimposed heatmap on the original image with prediction information

The visualization is saved to the specified output path or displayed if no output path is provided.

## Notes

- For the Traditional CNN model, the script uses the actual feature maps from the last convolutional layer to generate the heatmap.
- For the EfficientNetV2-S model, due to its complex architecture, the script uses a simplified approach to generate the heatmap.
- The script automatically detects the model type based on the model path and adjusts the preprocessing accordingly.
- The script requires TensorFlow, OpenCV, PIL, and Matplotlib to be installed.

## Troubleshooting

If you encounter any issues with the script, try the following:
- Make sure the model file exists and is a valid TensorFlow model.
- Make sure the image file exists and is a valid image file.
- Make sure the output directory exists or can be created.
- If you get an error about the model input shape, try using a different image or resizing the image to match the model's expected input shape.

## Requirements

Make sure you have the following dependencies installed:

```bash
pip install tensorflow opencv-python matplotlib numpy
```

## Directory Structure

The repository is organized as follows:

```
.
├── model_1_cnn/
│   └── Leukemia_Detection_CNN/
│       ├── models/
│       │   └── saved_models/
│       │       └── leukemia_model.h5  # Traditional CNN model
│       ├── scripts/
│       │   └── grad_cam_visualization.py  # Grad-CAM script for Traditional CNN
│       └── results/
│           └── grad_cam_visualizations/
│               └── traditional_cnn/  # Output directory for Traditional CNN
├── model_2_EfficientNetV2_s_cnn/
│   └── Leukemia_Detection_CNN/
│       ├── models/
│       │   └── efficientnetv2s_leukemia_model.h5  # EfficientNetV2-S model
│       ├── scripts/
│       │   └── grad_cam_visualization.py  # Grad-CAM script for EfficientNetV2-S
│       └── results/
│           └── grad_cam_visualizations/
│               └── efficientnetv2s/  # Output directory for EfficientNetV2-S
├── run_gradcam_visualization.py  # Main script to run both visualizations
└── README_gradcam.md  # This file
```

## Usage

### Running Both Models

To generate Grad-CAM visualizations for both models using sample images from the dataset:

```bash
python run_gradcam_visualization.py
```

By default, this will:
1. Look for sample images in the `model_1_cnn/Leukemia_Detection_CNN/data/ALL_dataset` directory
2. Select 3 samples from each class (Benign, Early, Pre, Pro)
3. Generate Grad-CAM visualizations for both models
4. Save the results in their respective output directories

### Custom Options

You can customize the behavior with the following options:

```bash
# Specify a different dataset directory
python run_gradcam_visualization.py --data_dir path/to/dataset

# Change the number of samples per class
python run_gradcam_visualization.py --num_samples 5

# Provide specific image paths
python run_gradcam_visualization.py --images path/to/image1.jpg path/to/image2.jpg
```

### Running Individual Models

You can also run the Grad-CAM visualization for each model separately:

#### Traditional CNN

```bash
cd model_1_cnn/Leukemia_Detection_CNN/scripts
python grad_cam_visualization.py path/to/image1.jpg path/to/image2.jpg
```

#### EfficientNetV2-S

```bash
cd model_2_EfficientNetV2_s_cnn/Leukemia_Detection_CNN/scripts
python grad_cam_visualization.py path/to/image1.jpg path/to/image2.jpg
```

## Output

For each input image, the scripts generate a visualization with three panels:
1. Original image
2. Grad-CAM heatmap
3. Superimposed heatmap on the original image with prediction information

The visualizations are saved in the respective output directories:
- Traditional CNN: `model_1_cnn/Leukemia_Detection_CNN/results/grad_cam_visualizations/traditional_cnn/`
- EfficientNetV2-S: `model_2_EfficientNetV2_s_cnn/Leukemia_Detection_CNN/results/grad_cam_visualizations/efficientnetv2s/`

## Interpretation

The Grad-CAM heatmaps highlight the regions that influenced the model's classification decision:
- Red areas indicate regions of high importance
- Blue areas indicate regions of low importance

For leukemia cell classification:
- **Benign cells**: The model should focus on the regular, round nucleus
- **Early-stage cells**: The model should attend to the irregular nuclear boundaries
- **Pre-cancerous cells**: The model should highlight the increased nuclear-to-cytoplasmic ratio
- **Progressive cells**: The model should focus on the highly irregular nuclear morphology and chromatin patterns

## References

- Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
- EfficientNetV2: Smaller Models and Faster Training
- Acute Lymphoblastic Leukemia (ALL) Dataset 