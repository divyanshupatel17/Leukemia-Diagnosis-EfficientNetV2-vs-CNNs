# config.py
IMAGE_SIZE = (224, 224)  # Increased image size
MODEL_PATH = "models/saved_models/efficientnetv2s_leukemia_model.h5"
CLASS_NAMES = ["Benign", "Early", "Pre", "Pro"]  # ALL_dataset classes
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 20  # Increased epochs
LEARNING_RATE = 0.0001  # Reduced learning rate
NUM_CLASSES = 4
DATA_DIR = "data/ALL_dataset"
MODEL_DATA_DIR = "data/efficientnet_model"  # Model-specific data directory