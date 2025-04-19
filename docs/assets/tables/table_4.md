# Table IV: Traditional CNN Model Architecture Details

| Layer | Output Shape | Parameters | Description |
|-------|-------------|------------|-------------|
| Input Layer | (128, 128, 3) | 0 | RGB input image with smaller dimensions |
| Conv2D | (128, 128, 32) | 896 | First convolutional layer with 3×3 kernels and ReLU |
| MaxPooling2D | (64, 64, 32) | 0 | 2×2 pooling to reduce dimensions |
| Conv2D | (64, 64, 64) | 18,496 | Second convolutional layer with 3×3 kernels and ReLU |
| MaxPooling2D | (32, 32, 64) | 0 | 2×2 pooling to reduce dimensions |
| Conv2D | (32, 32, 128) | 73,856 | Third convolutional layer with 3×3 kernels and ReLU |
| MaxPooling2D | (16, 16, 128) | 0 | 2×2 pooling to reduce dimensions |
| Flatten | (32,768) | 0 | Conversion to 1D vector |
| Dense | (128) | 4,194,432 | Fully connected layer with ReLU activation |
| Dropout | (128) | 0 | Rate = 0.5 for regularization |
| Dense | (4) | 516 | Output layer with softmax activation |
| Total Parameters | | 4,288,196 | ~4.3M trainable |
