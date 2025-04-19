# Table VII: Evaluation Metrics Used in the Study

| Metric | Formula | Description |
|--------|---------|-------------|
| Accuracy | Number of correct predictions / Total predictions | Overall model performance across all classes |
| Precision (class c) | TPc / (TPc + FPc) | Fraction of correctly identified positive instances |
| Recall (class c) | TPc / (TPc + FNc) | Fraction of correctly identified positive instances among all actual positive instances |
| F1-Score (class c) | 2 × (Precisionc × Recallc) / (Precisionc + Recallc) | Harmonic mean of precision and recall |
| Confusion Matrix | N/A | Visualization of prediction distribution across classes |
| ROC Curve (AUC) | integral from 0 to 1 of TPRc(FPRc-1(t))dt | One-vs-rest approach for multi-class |
