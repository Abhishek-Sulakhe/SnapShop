import os
import sys
import torch
import pandas as pd
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from shopee_matching.config import Config
from shopee_matching.train.train_image import train_image_model
from shopee_matching.train.train_text import train_text_model
from shopee_matching.train.train_multimodal import train_multimodal_model
from shopee_matching.train.train_lgb import train_lgb_model

def main():
    print("=====================================================")
    print("   Shopee Product Matching - Local Training Pipeline")
    print("=====================================================")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Input Dir: {Config.INPUT_DIR}")
    print(f"Output Dir: {Config.OUTPUT_DIR}")
    print("=====================================================")
    
    # Ensure directories exist
    Config.setup()
    
    # 1. Train Image Model
    print("\n[Stage 1/4] Training Image Model...")
    try:
        train_image_model()
    except Exception as e:
        print(f"Error in Image Training: {e}")
        # Continue? Or Exit? Let's continue to see if other parts work 
        # but usually we should stop.
    
    # 2. Train Text Model
    print("\n[Stage 2/4] Training Text Model...")
    try:
        train_text_model()
    except Exception as e:
        print(f"Error in Text Training: {e}")

    # 3. Train Multimodal Model
    print("\n[Stage 3/4] Training Multimodal Model...")
    try:
        train_multimodal_model()
    except Exception as e:
        print(f"Error in Multimodal Training: {e}")

    # 4. Train LightGBM
    # This Stage depends on the outputs of Stage 1 & 2 usually (embeddings)
    print("\n[Stage 4/4] Training LightGBM Reranker...")
    try:
        train_lgb_model()
    except Exception as e:
        print(f"Error in LightGBM Training: {e}")

    print("\n=====================================================")
    print("   Training Pipeline Complete")
    print("=====================================================")

if __name__ == '__main__':
    main()
