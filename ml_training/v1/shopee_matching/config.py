
import os
from pathlib import Path

class Config:
    # Paths - Updated for new structure
    BASE_DIR = Path(__file__).resolve().parent.parent # Points to ml_training/
    PROJECT_ROOT = BASE_DIR.parent # Points to project root

    # Input/Output
    INPUT_DIR = PROJECT_ROOT / 'input'
    OUTPUT_DIR = PROJECT_ROOT / 'output'
    
    # Checkpoints
    CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'

    # Data
    TRAIN_CSV = INPUT_DIR / 'shopee-product-matching/train.csv'
    TRAIN_IMAGES = INPUT_DIR / 'shopee-product-matching/train_images'

    # Training
    SEED = 42
    IMG_SIZE = 512
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    DEVICE = 'cuda'
    
    # Model
    MODEL_NAME = 'tf_efficientnet_b0_ns'
    BERT_MODEL = 'bert-base-multilingual-cased'
    NUM_CLASSES = 11014
    
    EPOCHS = 15
    LR = 1e-4
    MIN_LR = 1e-6

    @classmethod
    def setup(cls):
        cls.INPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
