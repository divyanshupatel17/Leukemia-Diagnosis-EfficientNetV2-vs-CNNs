import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.patches as patches
import argparse

# Define colors for architecture diagram
BLOCK_COLORS = {
    'Input': '#E6F2FF',
    'Conv2D': '#FFD700',
    'BatchNormalization': '#98FB98',
    'MaxPooling2D': '#87CEFA',
    'Dropout': '#FFA07A',
    'Flatten': '#D8BFD8',
    'Dense': '#FF6347',
    'EfficientNetV2S': '#9370DB',
    'GlobalAveragePooling2D': '#20B2AA'
}

def create_detailed_architecture_diagram(model_path, output_path, is_efficientnet=False):
    """
    Create a detailed architecture diagram for the model.
    
    Args:
        model_path (str): Path to the model file.
        output_path (str): Path to save the architecture diagram.
        is_efficientnet (bool): Whether the model is EfficientNetV2-S.
    """
    # Load the model
    model = load_model(model_path)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 20))
    
    # Set title based on model type
    title = "EfficientNetV2-S CNN Architecture" if is_efficientnet else "Traditional CNN Architecture"
    ax.set_title(title, fontsize=22, fontweight='bold')
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set axis limits
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 120)
    
    # Define block dimensions
    block_width = 70
    block_height = 6
    x_start = 25
    y_start = 110
    y_spacing = 7
    
    # Get model layers
    layers = model.layers
    
    # Prepare blocks based on model type
    if is_efficientnet:
        # EfficientNetV2-S has a more complex architecture, so we'll create a more detailed representation
        blocks = [
            {'name': 'Input Layer', 'type': 'Input', 'details': '(224, 224, 3)', 'shape': '(None, 224, 224, 3)'},
            {'name': 'EfficientNetV2-S Base', 'type': 'EfficientNetV2S', 'details': 'Pre-trained on ImageNet', 'shape': '(None, 7, 7, 1280)'},
            {'name': 'Global Average Pooling', 'type': 'GlobalAveragePooling2D', 'details': 'Flatten features', 'shape': '(None, 1280)'},
            {'name': 'Dense Layer', 'type': 'Dense', 'details': '128 units, ReLU', 'shape': '(None, 128)'},
            {'name': 'Dropout', 'type': 'Dropout', 'details': 'rate=0.5', 'shape': '(None, 128)'},
            {'name': 'Output Layer', 'type': 'Dense', 'details': '4 units, Softmax', 'shape': '(None, 4)'}
        ]
        
        # Add EfficientNetV2-S internal architecture details
        efficientnet_blocks = [
            {'name': 'Stem', 'type': 'Conv2D', 'details': '24 filters, 3x3, stride=2', 'shape': '(None, 112, 112, 24)'},
            {'name': 'MBConv Block 1', 'type': 'Conv2D', 'details': '3 layers, expansion=1', 'shape': '(None, 112, 112, 24)'},
            {'name': 'MBConv Block 2', 'type': 'Conv2D', 'details': '5 layers, expansion=4', 'shape': '(None, 56, 56, 48)'},
            {'name': 'MBConv Block 3', 'type': 'Conv2D', 'details': '5 layers, expansion=4', 'shape': '(None, 28, 28, 64)'},
            {'name': 'MBConv Block 4', 'type': 'Conv2D', 'details': '7 layers, expansion=4', 'shape': '(None, 14, 14, 128)'},
            {'name': 'MBConv Block 5', 'type': 'Conv2D', 'details': '14 layers, expansion=6', 'shape': '(None, 14, 14, 160)'},
            {'name': 'MBConv Block 6', 'type': 'Conv2D', 'details': '18 layers, expansion=6', 'shape': '(None, 7, 7, 256)'},
            {'name': 'Top Conv', 'type': 'Conv2D', 'details': '1280 filters, 1x1', 'shape': '(None, 7, 7, 1280)'}
        ]
        
        # Insert EfficientNetV2-S details after the base block
        blocks = blocks[:2] + efficientnet_blocks + blocks[2:]
    else:
        # For traditional CNN, extract actual layer information
        blocks = []
        for layer in layers:
            layer_type = layer.__class__.__name__
            
            # Map layer type to our color scheme
            block_type = layer_type
            if layer_type not in BLOCK_COLORS:
                # Find the closest match
                for key in BLOCK_COLORS.keys():
                    if key in layer_type:
                        block_type = key
                        break
                else:
                    block_type = 'Other'
            
            # Get layer details
            config = layer.get_config()
            details = ""
            
            if layer_type == 'Conv2D':
                details = f"{config['filters']} filters, {config['kernel_size'][0]}x{config['kernel_size'][1]}, {config['activation']}"
            elif layer_type == 'Dense':
                details = f"{config['units']} units, {config['activation']}"
            elif layer_type == 'Dropout':
                details = f"rate={config['rate']}"
            elif layer_type == 'MaxPooling2D':
                details = f"{config['pool_size'][0]}x{config['pool_size'][1]}"
            
            # Get output shape
            shape = ""
            try:
                if hasattr(layer, 'output_shape'):
                    shape = str(layer.output_shape)
                elif hasattr(layer, 'output'):
                    shape = str(layer.output.shape)
            except:
                shape = "Shape unavailable"
            
            blocks.append({
                'name': layer.name,
                'type': block_type,
                'details': details,
                'shape': shape
            })
    
    # Draw blocks
    for i, block in enumerate(blocks):
        y_pos = y_start - i * y_spacing
        
        # Get color based on block type
        color = BLOCK_COLORS.get(block['type'], '#CCCCCC')
        
        # Draw block
        rect = patches.Rectangle((x_start, y_pos), block_width, block_height, 
                                linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        
        # Add block name and details
        block_text = f"{block['name']}"
        if block['details']:
            block_text += f"\n{block['details']}"
        
        # Add output shape
        if 'shape' in block:
            block_text += f"\nOutput: {block['shape']}"
        
        ax.text(x_start + block_width/2, y_pos + block_height/2, block_text,
                ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Add arrow to next block (if not the last block)
        if i < len(blocks) - 1:
            ax.arrow(x_start + block_width/2, y_pos, 0, -y_spacing + block_height,
                    head_width=2, head_length=1, fc='black', ec='black')
    
    # Add legend
    legend_x = 10
    legend_y = 30
    legend_spacing = 4
    legend_size = 3
    
    # Draw legend title
    ax.text(legend_x, legend_y + 4, "Layer Types:", fontsize=14, fontweight='bold')
    
    # Draw legend items
    for i, (layer_type, color) in enumerate(BLOCK_COLORS.items()):
        y_legend = legend_y - i * legend_spacing
        
        # Draw legend box
        rect = patches.Rectangle((legend_x, y_legend), legend_size, legend_size, 
                                linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        
        # Add legend text
        ax.text(legend_x + legend_size + 2, y_legend + legend_size/2, layer_type,
                va='center', fontsize=10)
    
    # Add model summary as text
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    
    # Join summary lines and add to figure
    summary_text = '\n'.join(summary_lines)
    
    # Add text box with model summary
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(105, 60, summary_text, fontsize=7, family='monospace',
            verticalalignment='center', bbox=props, transform=ax.transData)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Architecture diagram saved to {output_path}")
    plt.close()

def create_combined_architecture_figure(output_path):
    """
    Create a combined figure showing both model architectures side by side.
    
    Args:
        output_path (str): Path to save the combined figure.
    """
    # Create figure with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 16))
    
    # Set titles
    ax1.set_title("Traditional CNN Architecture", fontsize=20, fontweight='bold')
    ax2.set_title("EfficientNetV2-S CNN Architecture", fontsize=20, fontweight='bold')
    
    # Remove axis ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Set axis limits
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    
    # Define block dimensions
    block_width = 60
    block_height = 5
    x_start = 20
    y_start = 95
    y_spacing = 6
    
    # Draw Traditional CNN architecture
    cnn_blocks = [
        {'name': 'Input Layer', 'type': 'Input', 'details': '(128, 128, 3)'},
        {'name': 'Conv2D', 'type': 'Conv2D', 'details': '32 filters, 3x3, ReLU'},
        {'name': 'BatchNormalization', 'type': 'BatchNormalization', 'details': ''},
        {'name': 'MaxPooling2D', 'type': 'MaxPooling2D', 'details': '2x2'},
        {'name': 'Conv2D', 'type': 'Conv2D', 'details': '64 filters, 3x3, ReLU'},
        {'name': 'BatchNormalization', 'type': 'BatchNormalization', 'details': ''},
        {'name': 'MaxPooling2D', 'type': 'MaxPooling2D', 'details': '2x2'},
        {'name': 'Conv2D', 'type': 'Conv2D', 'details': '128 filters, 3x3, ReLU'},
        {'name': 'BatchNormalization', 'type': 'BatchNormalization', 'details': ''},
        {'name': 'MaxPooling2D', 'type': 'MaxPooling2D', 'details': '2x2'},
        {'name': 'Flatten', 'type': 'Flatten', 'details': ''},
        {'name': 'Dense', 'type': 'Dense', 'details': '128 units, ReLU'},
        {'name': 'Dropout', 'type': 'Dropout', 'details': 'rate=0.5'},
        {'name': 'Output Layer', 'type': 'Dense', 'details': '4 units, Softmax'}
    ]
    
    # Draw blocks for Traditional CNN
    for i, block in enumerate(cnn_blocks):
        y_pos = y_start - i * y_spacing
        
        # Get color based on block type
        color = BLOCK_COLORS.get(block['type'], '#CCCCCC')
        
        # Draw block
        rect = patches.Rectangle((x_start, y_pos), block_width, block_height, 
                                linewidth=1, edgecolor='black', facecolor=color)
        ax1.add_patch(rect)
        
        # Add block name and details
        block_text = f"{block['name']}"
        if block['details']:
            block_text += f"\n{block['details']}"
        
        ax1.text(x_start + block_width/2, y_pos + block_height/2, block_text,
                ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Add arrow to next block (if not the last block)
        if i < len(cnn_blocks) - 1:
            ax1.arrow(x_start + block_width/2, y_pos, 0, -y_spacing + block_height,
                    head_width=2, head_length=1, fc='black', ec='black')
    
    # Draw EfficientNetV2-S architecture
    efficientnet_blocks = [
        {'name': 'Input Layer', 'type': 'Input', 'details': '(224, 224, 3)'},
        {'name': 'EfficientNetV2-S Base', 'type': 'EfficientNetV2S', 'details': 'Pre-trained on ImageNet'},
        {'name': 'Stem', 'type': 'Conv2D', 'details': '24 filters, 3x3, stride=2'},
        {'name': 'MBConv Block 1', 'type': 'Conv2D', 'details': '3 layers, expansion=1'},
        {'name': 'MBConv Block 2', 'type': 'Conv2D', 'details': '5 layers, expansion=4'},
        {'name': 'MBConv Block 3', 'type': 'Conv2D', 'details': '5 layers, expansion=4'},
        {'name': 'MBConv Block 4', 'type': 'Conv2D', 'details': '7 layers, expansion=4'},
        {'name': 'MBConv Block 5', 'type': 'Conv2D', 'details': '14 layers, expansion=6'},
        {'name': 'MBConv Block 6', 'type': 'Conv2D', 'details': '18 layers, expansion=6'},
        {'name': 'Top Conv', 'type': 'Conv2D', 'details': '1280 filters, 1x1'},
        {'name': 'Global Average Pooling', 'type': 'GlobalAveragePooling2D', 'details': 'Flatten features'},
        {'name': 'Dense Layer', 'type': 'Dense', 'details': '128 units, ReLU'},
        {'name': 'Dropout', 'type': 'Dropout', 'details': 'rate=0.5'},
        {'name': 'Output Layer', 'type': 'Dense', 'details': '4 units, Softmax'}
    ]
    
    # Draw blocks for EfficientNetV2-S
    for i, block in enumerate(efficientnet_blocks):
        y_pos = y_start - i * y_spacing
        
        # Get color based on block type
        color = BLOCK_COLORS.get(block['type'], '#CCCCCC')
        
        # Draw block
        rect = patches.Rectangle((x_start, y_pos), block_width, block_height, 
                                linewidth=1, edgecolor='black', facecolor=color)
        ax2.add_patch(rect)
        
        # Add block name and details
        block_text = f"{block['name']}"
        if block['details']:
            block_text += f"\n{block['details']}"
        
        ax2.text(x_start + block_width/2, y_pos + block_height/2, block_text,
                ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Add arrow to next block (if not the last block)
        if i < len(efficientnet_blocks) - 1:
            ax2.arrow(x_start + block_width/2, y_pos, 0, -y_spacing + block_height,
                    head_width=2, head_length=1, fc='black', ec='black')
    
    # Add legend
    legend_x = 10
    legend_y = 20
    legend_spacing = 4
    legend_size = 3
    
    # Draw legend title
    fig.text(0.05, 0.95, "Layer Types:", fontsize=14, fontweight='bold')
    
    # Draw legend items in a horizontal layout at the top
    for i, (layer_type, color) in enumerate(BLOCK_COLORS.items()):
        x_legend = 0.05 + i * 0.11
        
        # Draw legend box
        fig.text(x_legend, 0.93, "■", fontsize=16, color=color)
        
        # Add legend text
        fig.text(x_legend + 0.02, 0.93, layer_type, fontsize=10)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust for the legend at the top
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined architecture diagram saved to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Create detailed architecture diagrams for CNN models")
    parser.add_argument("--cnn_model", type=str, default="model_1_cnn/Leukemia_Detection_CNN/models/saved_models/leukemia_model.h5",
                        help="Path to the Traditional CNN model file")
    parser.add_argument("--efficientnet_model", type=str, default="model_2_EfficientNetV2_s_cnn/Leukemia_Detection_CNN/models/efficientnetv2s_leukemia_model.h5",
                        help="Path to the EfficientNetV2-S model file")
    parser.add_argument("--output_dir", type=str, default="figures", help="Directory to save the figures")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create detailed architecture diagrams
    try:
        print("Creating Traditional CNN architecture diagram...")
        create_detailed_architecture_diagram(
            args.cnn_model,
            os.path.join(args.output_dir, "traditional_cnn_architecture_detailed.png"),
            is_efficientnet=False
        )
        print("Traditional CNN architecture diagram created successfully.")
    except Exception as e:
        print(f"Error creating Traditional CNN architecture diagram: {str(e)}")
    
    try:
        print("Creating EfficientNetV2-S architecture diagram...")
        create_detailed_architecture_diagram(
            args.efficientnet_model,
            os.path.join(args.output_dir, "efficientnetv2s_architecture_detailed.png"),
            is_efficientnet=True
        )
        print("EfficientNetV2-S architecture diagram created successfully.")
    except Exception as e:
        print(f"Error creating EfficientNetV2-S architecture diagram: {str(e)}")
    
    # Create combined architecture figure
    try:
        print("Creating combined architecture diagram...")
        create_combined_architecture_figure(
            os.path.join(args.output_dir, "combined_model_architectures.png")
        )
        print("Combined architecture diagram created successfully.")
    except Exception as e:
        print(f"Error creating combined architecture diagram: {str(e)}")

if __name__ == "__main__":
    main() 