import os
import sys
import argparse
import tensorflow as tf

def test_model_loading(model_path):
    """
    Test if a model file can be loaded by TensorFlow.
    
    Args:
        model_path (str): Path to the model file.
        
    Returns:
        bool: True if the model can be loaded, False otherwise.
    """
    print(f"Testing model loading for: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    try:
        print("Attempting to load model...")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        print(f"Model summary:")
        model.summary()
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test if a model file can be loaded by TensorFlow")
    parser.add_argument("model_path", type=str, help="Path to the model file")
    
    args = parser.parse_args()
    
    # Test model loading
    success = test_model_loading(args.model_path)
    
    if success:
        print("\nModel is valid and can be loaded by TensorFlow.")
        sys.exit(0)
    else:
        print("\nModel could not be loaded. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 