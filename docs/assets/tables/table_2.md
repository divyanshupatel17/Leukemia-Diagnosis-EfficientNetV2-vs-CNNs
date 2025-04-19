# Table II: Image Preprocessing Steps for the ALL Dataset

| Preprocessing Step | EfficientNetV2-S | Traditional CNN | Description |
|-------------------|-------------------|-----------------|-------------|
| Image Resizing | 224×224 pixels | 128×128 pixels | Standardizes input dimensions for model compatibility |
| Normalization | [0,1] range | [0,1] range | Scales pixel values by dividing by 255 |
| Color Format | RGB (3 channels) | RGB (3 channels) | Maintains color information for feature extraction |
| Data Augmentation | Rotation (±15°) | Rotation (±15°) | Enhances dataset variety Horizontal Flip Vertical Flip Zoom (±10%) Brightness Adjustment (±10%) |
| Data Splitting | 60% Training | 60% Training | Allocates data for different phases 20% Validation 20% Testing |
| Stratification | Applied | Applied | Maintains class distribution across splits |
