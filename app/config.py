# Configuration parameters

EPOCHS = 20
LR = 0.1
STEP = 15
GAMMA = 0.1
BATCH = 32
OUT_CLASSES = 3
IMG_SIZE = 224

# Data paths
FACES_PATH = "data/preprocessed_faces_mediapipe.npy"
LABELS_PATH = "data/preprocessed_labels_mediapipe.npy"

# Labels
LABEL_INDEX = {"dry": 0, "normal": 1, "oily": 2}
INDEX_LABEL = {0: "dry", 1: "normal", 2: "oily"}

# Device configuration
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model saving path
MODEL_SAVE_PATH = "model_checkpoints/"