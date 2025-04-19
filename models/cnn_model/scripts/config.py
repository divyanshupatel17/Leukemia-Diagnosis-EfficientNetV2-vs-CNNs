# config.py
IMAGE_SIZE = (128, 128)  # Or (224, 224) - Adjust as needed
MODEL_PATH = "models/saved_models/leukemia_model.h5"
NUM_CLASSES = 4  # Benign, Early, Pre, Pro,
TEST_SIZE = 0.2  # 20% for testing
VALIDATION_SPLIT = 0.2 # Percentage of train data for validation
RANDOM_STATE = 42  # For reproducibility
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
CLASS_NAMES = ["Benign", "Early", "Pre", "Pro"]  # ALL_dataset classes
DATA_DIR = "data/ALL_dataset"  # Directory for original data
MODEL_DATA_DIR = "data/cnn_model"  # Model-specific data directory
