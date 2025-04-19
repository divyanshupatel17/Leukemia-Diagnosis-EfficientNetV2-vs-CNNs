# Table III: Enhanced EfficientNetV2-S Model Architecture Details

| Layer | Output Shape | Parameters | Description |
|-------|-------------|------------|-------------|
| Input Layer | (224, 224, 3) | 0 | RGB input image |
| EfficientNetV2-S Backbone | Variable | ~20.7M | Pre-trained on ImageNet with selective layer freezing |
| Global Average Pooling | (1280) | 0 | Reduces spatial dimensions while preserving feature map integrity |
| Dense Layer | (128) | 163,968 | Bottleneck representation with ReLU activation |
| Dropout Layer | (128) | 0 | Rate = 0.5 for regularization to prevent overfitting |
| Output Layer | (4) | 516 | Softmax activation for 4 class probabilities |
| Total Parameters | | 20.9M | ~20.1M trainable |
