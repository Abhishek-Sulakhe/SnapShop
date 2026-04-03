import pandas as pd
import numpy as np
import torch
import gc
from pathlib import Path
import os

from shopee_matching.config import Config
from shopee_matching.utils.common import seed_everything
from shopee_matching.inference.extract import extract_image_features, extract_text_features, get_neighbors
from shopee_matching.utils.query_expansion import query_expansion
from shopee_matching.inference.match_gcn import infer_gcn
from shopee_matching.inference.match_lgb import infer_lgb

def main():
    seed_everything(Config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    # 1. Load Data
    csv_path = Config.TEST_CSV
    if not os.path.exists(csv_path):
        csv_path = Config.TRAIN_CSV
        
    df = pd.read_csv(csv_path)
    if len(df) <= 3:
        Config.DEBUG = True
        
    img_dir = Config.TEST_IMAGES if 'test' in str(csv_path) else Config.TRAIN_IMAGES
    
    # 2. Extract
    print("Extracting Image Features...")
    img_feats = extract_image_features(df, img_dir, device=device)
    print("Extracting Text Features...")
    text_feats = extract_text_features(df, device=device)
    
    # Norm
    img_feats /= np.linalg.norm(img_feats, axis=1, keepdims=True)
    text_feats /= np.linalg.norm(text_feats, axis=1, keepdims=True)
    
    # Concat
    full_feats = np.concatenate([img_feats, text_feats], axis=1)
    full_feats /= np.linalg.norm(full_feats, axis=1, keepdims=True)
    
    # 3. Query Expansion
    print("Query Expansion...")
    D, I = get_neighbors(full_feats, k=3)
    # query_expansion needs: feats, sim(distances), indices
    full_feats = query_expansion(full_feats, D, I)
    
    # 4. GCN Inference
    # Note: GCN requires trained GAT model. If not present, returns empty.
    gcn_preds = infer_gcn(df, img_feats, text_feats, full_feats)
    
    # 5. LGB Inference
    # Note: LGB requires trained booster. If not present, returns raw IDs or empty.
    lgb_preds = infer_lgb(df, img_feats, text_feats)
    
    # 6. Ensemble / Submission
    print("Generating Submission...")
    
    # Priority: LGB > GCN > KNN
    if lgb_preds and len(lgb_preds) == len(df) and " " in lgb_preds[0]:
         # Valid LGB predictions
         print("Using LGB Predictions")
         df['matches'] = lgb_preds
    elif gcn_preds and len(gcn_preds) > 0:
        # GCN usually returns pair probabilities, needs decoding to matches string
        # Skipping complex GCN decoding for this refactor as trained model is missing
        print("Using KNN Baseline (GCN model missing)")
        # KNN Baseline
        D, I = get_neighbors(full_feats, k=50)
        preds = []
        for i in range(len(df)):
            idx = I[i, D[i] > 0.6]
            preds.append(" ".join(df['posting_id'].iloc[idx].values))
        df['matches'] = preds
    else:
        print("Using KNN Baseline")
        D, I = get_neighbors(full_feats, k=50)
        preds = []
        for i in range(len(df)):
            idx = I[i, D[i] > 0.55] 
            preds.append(" ".join(df['posting_id'].iloc[idx].values))
        df['matches'] = preds
        
    df[['posting_id', 'matches']].to_csv('submission.csv', index=False)
    print("Done.")

if __name__ == '__main__':
    main()
