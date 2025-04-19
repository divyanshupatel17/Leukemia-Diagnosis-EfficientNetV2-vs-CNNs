"""
Model Comparison Utility

This script compares the performance of the Traditional CNN and Enhanced EfficientNetV2-S 
models by generating comparison plots and metrics tables.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add the project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_comparison_chart(cnn_metrics, efficientnet_metrics, output_path):
    """Create a bar chart comparing key metrics between models."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    cnn_values = [cnn_metrics[m.lower()] for m in metrics]
    efficientnet_values = [efficientnet_metrics[m.lower()] for m in metrics]
    
    x = np.arange(len(metrics))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x - width/2, cnn_values, width, label='Traditional CNN')
    ax.bar(x + width/2, efficientnet_values, width, label='Enhanced EfficientNetV2-S')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Model Performance Comparison', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.1)  # Set y-axis to start at 0 and end at 1.1 for better visualization
    
    # Add value labels on top of each bar
    for i, v in enumerate(cnn_values):
        ax.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
    
    for i, v in enumerate(efficientnet_values):
        ax.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
    
    ax.legend(fontsize=12)
    
    # Add a grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved comparison chart to {output_path}")
    plt.close()

def create_radar_chart(cnn_metrics, efficientnet_metrics, output_path):
    """Create a radar chart comparing metrics between models."""
    # Metrics for the radar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Convert metrics to values
    cnn_values = [cnn_metrics[m.lower()] for m in metrics]
    efficientnet_values = [efficientnet_metrics[m.lower()] for m in metrics]
    
    # Number of variables
    N = len(metrics)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    
    # Make the plot close by repeating the first value
    cnn_values += [cnn_values[0]]
    efficientnet_values += [efficientnet_values[0]]
    angles += [angles[0]]
    metrics += [metrics[0]]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Draw the lines and points for CNN model
    ax.plot(angles, cnn_values, 'o-', linewidth=2, label='Traditional CNN')
    ax.fill(angles, cnn_values, alpha=0.25)
    
    # Draw the lines and points for EfficientNet model
    ax.plot(angles, efficientnet_values, 'o-', linewidth=2, label='Enhanced EfficientNetV2-S')
    ax.fill(angles, efficientnet_values, alpha=0.25)
    
    # Set the labels
    ax.set_thetagrids(np.degrees(angles), metrics)
    
    # Draw the y-axis labels (0.2, 0.4, etc.)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    
    # Add a title
    ax.set_title("Model Performance Metrics Comparison", fontsize=16)
    
    # Add a legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved radar chart to {output_path}")
    plt.close()

def create_comparison_table(cnn_metrics, efficientnet_metrics, output_path):
    """Create a CSV table comparing all metrics between models."""
    # Combine metrics into a DataFrame
    metrics_df = pd.DataFrame({
        'Metric': list(cnn_metrics.keys()),
        'Traditional CNN': list(cnn_metrics.values()),
        'Enhanced EfficientNetV2-S': list(efficientnet_metrics.values())
    })
    
    # Save to CSV
    metrics_df.to_csv(output_path, index=False)
    print(f"Saved comparison table to {output_path}")
    
    return metrics_df

def load_metrics(model_name):
    """Load the metrics for a given model."""
    if model_name == "cnn":
        metrics_path = os.path.join("models", "cnn_model", "results", "tables", "metrics_summary.json")
    else:  # efficientnet
        metrics_path = os.path.join("models", "efficientnet_model", "results", "tables", "metrics_summary.json")
    
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    except FileNotFoundError:
        print(f"Metrics file not found: {metrics_path}")
        # Return dummy metrics for testing
        return {
            "accuracy": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "f1-score": 0.5
        }

def main():
    """Main function to run the model comparison."""
    print("Starting model comparison...")
    
    # Create output directories if they don't exist
    output_dir = os.path.join("docs", "assets", "figures", "model_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics for both models
    cnn_metrics = load_metrics("cnn")
    efficientnet_metrics = load_metrics("efficientnet")
    
    # Create comparison outputs
    bar_chart_path = os.path.join(output_dir, "performance_comparison_bar.png")
    radar_chart_path = os.path.join(output_dir, "performance_comparison_radar.png")
    table_path = os.path.join(output_dir, "performance_comparison.csv")
    
    # Generate the comparison artifacts
    create_comparison_chart(cnn_metrics, efficientnet_metrics, bar_chart_path)
    create_radar_chart(cnn_metrics, efficientnet_metrics, radar_chart_path)
    df = create_comparison_table(cnn_metrics, efficientnet_metrics, table_path)
    
    # Print the comparison table to console
    print("\nModel Performance Comparison:")
    print(df.to_string(index=False))
    
    print("\nModel comparison completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 