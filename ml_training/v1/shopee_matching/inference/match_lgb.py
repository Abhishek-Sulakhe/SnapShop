import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
from tqdm import tqdm
import Levenshtein
import glob
from shopee_matching.config import Config

def get_neighbors(feats, k=50):
    try:
        import faiss.contrib.torch_utils
    except ImportError:
        pass
    d = feats.shape[1]
    index = faiss.IndexFlatIP(d)
    if torch.cuda.is_available():
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except:
            pass
    index.add(feats)
    D, I = index.search(feats, k)
    return D, I

def infer_lgb(df, img_embeddings, text_embeddings):
    # Check for models
    model_files = glob.glob(str(Config.CHECKPOINT_DIR / 'lgb_fold_*.txt'))
    if not model_files:
        print("No LGB models found.")
        return [f"{pid}" for pid in df['posting_id']]

    print(f"Found {len(model_files)} LGB models.")
    
    # Normalize 
    img_embeddings = img_embeddings / np.linalg.norm(img_embeddings, axis=1, keepdims=True)
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    
    # Neighbors. Need import faiss here or import correctly
    import faiss
    
    try:
        D_img, I_img = get_neighbors(img_embeddings, k=50)
        D_text, I_text = get_neighbors(text_embeddings, k=50)
    except Exception as e:
        print(f"KNN Failed: {e}")
        return [f"{pid}" for pid in df['posting_id']]
        
    titles = df['title'].values
    predictions = []
    pid_map = df['posting_id'].values
    
    models = [lgb.Booster(model_file=f) for f in model_files]
    
    # Batch processing or row-by-row
    # Row-by-row for simplicity as in training logic
    
    for i in tqdm(range(len(df)), desc="LGB Predicting"):
        valid_indices = set(I_img[i]) | set(I_text[i])
        valid_indices.discard(i)
        candidates = list(valid_indices)
        
        if not candidates:
            predictions.append(pid_map[i])
            continue
            
        feats = []
        for j in candidates:
            # Recalculate features
            sim_img = np.dot(img_embeddings[i], img_embeddings[j])
            sim_text = np.dot(text_embeddings[i], text_embeddings[j])
            if isinstance(titles[i], str) and isinstance(titles[j], str):
                dist = Levenshtein.distance(titles[i], titles[j])
            else:
                dist = 999
            feats.append([sim_img, sim_text, dist])
            
        feats = np.array(feats)
        
        preds = np.zeros(len(feats))
        for model in models:
            preds += model.predict(feats)
        preds /= len(models)
        
        matches = [candidates[k] for k in range(len(preds)) if preds[k] > 0.6]
        matches.append(i)
        
        match_ids = pid_map[matches]
        predictions.append(" ".join(match_ids))
        
    return predictions
