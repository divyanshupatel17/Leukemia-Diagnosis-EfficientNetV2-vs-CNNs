# Table VI: Hyperparameter Configuration for Model Training

| Hyperparameter | EfficientNetV2-S | Traditional CNN | Description |
|----------------|-------------------|-----------------|-------------|
| Image Size | 224 × 224 pixels | 128 × 128 pixels | Input size for each model |
| Batch Size | 32 | 32 | Balances computational efficiency and gradient estimation |
| Epochs | 20 | 20 | Number of complete passes through the dataset |
| Learning Rate | 0.0001 | 0.001 | Initial learning rate for optimization |
| Learning Rate Decay | alpha(t)=alpha(0)/(1+k*t) | alpha(t)=alpha(0)/(1+k*t) | Decay factor k applied over time t |
| Optimizer | Adam | Adam | Adaptive learning rate optimization algorithm |
| Loss Function | Categorical Cross-Entropy | Categorical Cross-Entropy | Suitable for multi-class classification tasks |
| Dropout Rate | 0.5 | 0.5 | Regularization applied in the classification head |
| Early Stopping Patience | 10 | 10 | Stops training after specified epochs with no improvement |
| Class Weights | Inversely proportional to frequencies | Inversely proportional to frequencies | Addresses class imbalance |
| Random State | 42 | 42 | Ensures reproducibility of results |
