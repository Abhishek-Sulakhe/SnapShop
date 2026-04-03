import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import gc
import torch
import os
import Levenshtein

from shopee_matching.config import Config
from shopee_matching.inference.extract import extract_image_features, extract_text_features, get_neighbors

def get_text_diff(text, text_base):
    if not isinstance(text, str) or not isinstance(text_base, str):
        return 999
    return Levenshtein.distance(text, text_base)

def train_lgb_model():
    if not os.path.exists(Config.TRAIN_CSV):
        print("Train CSV not found.")
        return

    df = pd.read_csv(Config.TRAIN_CSV)
    
    # 1. Feature Extraction (Using the models we potentially just trained, or pretrained)
    # Check if we have trained models in CHECKPOINT_SAVE_DIR, else use defaults from Config
    img_ckpt = Config.SHOPEE_MODEL
    if not img_ckpt.exists():
        print(f"Propagated image checkpoint not found at {img_ckpt}. Training might fail/use random weights if not handled in extract.")
        # extract_image_features handles None by creating fresh model (random weights) -> bad for LGB
        # But we assume user ran train.py first.
        
    text_ckpt = Config.BERT_MODEL
    if not text_ckpt.exists():
        print(f"Propagated text checkpoint not found at {text_ckpt}.")

    print(f"Extracting features from Train data... (Size: {len(df)})")
    
    # Check if features are already saved to save time
    # cache_path = Config.CHECKPOINT_DIR / 'train_feats.pth'
    
    image_embeddings = extract_image_features(df, Config.TRAIN_IMAGES, checkpoint_path=img_ckpt)
    text_embeddings = extract_text_features(df, checkpoint_path=text_ckpt)
    
    # Normalize
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    
    # Concat (Simple concatenation for candidate generation, or separate)
    # Notebook strategy: Generate candidates from Image KNN + Text KNN
    
    print("Finding neighbors...")
    # Image Neighbors
    distances_img, indices_img = get_neighbors(image_embeddings, k=50)
    # Text Neighbors
    distances_text, indices_text = get_neighbors(text_embeddings, k=50)
    
    # 2. Candidate Generation & Labeling
    print("Generating pairs and features...")
    
    # We will process in chunks or iterate rows
    # For training, we need negative samples too.
    # The neighbors serve as hard negatives.
    
    pairs = []
    labels = []
    
    # Features
    cosine_sims = []
    text_dists = []
    
    label_groups = df['label_group'].values
    titles = df['title'].values
    
    for i in tqdm(range(len(df))):
        # Union of neighbors
        candidates = set(indices_img[i]) | set(indices_text[i])
        candidates.discard(i) # Remove self? Or keep self as positive?
        # Usually we keep self or remove, let's keep self for recall calculation but for binary clf maybe remove
        # Let's remove self to focus on finding *other* matches
        
        target_group = label_groups[i]
        
        for j in candidates:
            # Pair (i, j)
            # Label
            is_match = 1 if label_groups[j] == target_group else 0
            
            # Features
            # Sim
            # We need to find the sim from the distance matrices or recalculate
            # Recalculating dot product is fast for single pair
            sim_img = np.dot(image_embeddings[i], image_embeddings[j])
            sim_text = np.dot(text_embeddings[i], text_embeddings[j])
            
            # Text diff
            dist = get_text_diff(titles[i], titles[j])
            
            pairs.append((i, j))
            labels.append(is_match)
            cosine_sims.append([sim_img, sim_text])
            text_dists.append(dist)
            
    # Create DataFrame
    pair_df = pd.DataFrame(pairs, columns=['idx1', 'idx2'])
    pair_df['label'] = labels
    pair_df['sim_img'] = [x[0] for x in cosine_sims]
    pair_df['sim_text'] = [x[1] for x in cosine_sims]
    pair_df['text_dist'] = text_dists
    
    print(f"Generated {len(pair_df)} pairs. Positives: {pair_df['label'].sum()}")
    
    # 3. Train LGB
    X = pair_df[['sim_img', 'sim_text', 'text_dist']]
    y = pair_df['label']
    
    # GroupKFold (but we constructed flat pairs, we need 'group' info for split if we want strict leak prevention)
    # Actually, we should split by label_group of idx1
    groups = df.loc[pair_df['idx1'], 'label_group'].values
    
    kfold = GroupKFold(n_splits=5)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    print("Training LGBM...")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(params, dtrain, valid_sets=[dtrain, dval], 
                          num_boost_round=1000, 
                          callbacks=[
                              lgb.early_stopping(stopping_rounds=50),
                              lgb.log_evaluation(100)
                          ])
        
        # Save model
        save_path = Config.CHECKPOINT_SAVE_DIR / f'lgb_fold_{fold}.txt'
        model.save_model(str(save_path))
        print(f"Saved LGB model to {save_path}")
        
    print("LGB training finished.")

if __name__ == '__main__':
    train_lgb_model()
