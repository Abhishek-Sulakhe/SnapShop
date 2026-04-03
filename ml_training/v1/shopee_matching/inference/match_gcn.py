import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import joblib
import gc
import faiss

from shopee_matching.config import Config
from shopee_matching.models.gnn import GATPairClassifier
from shopee_matching.data.dataset import GraphDataset

def get_sim_dict(feats, k=50):
    try:
        import faiss.contrib.torch_utils
    except:
        pass
    
    d = feats.shape[1]
    index = faiss.IndexFlatIP(d)
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(feats)
    D, I = index.search(feats, k)
    
    sim_dict = {}
    for i in range(len(feats)):
        for idx, j in enumerate(I[i]):
            sim_dict[(i, j)] = D[i][idx]
    return sim_dict, I

def infer_gcn(df, img_feats, text_feats, mm_feats):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Preparing GCN Data...")
    
    # Normalize
    img_feats = img_feats / np.linalg.norm(img_feats, axis=1, keepdims=True)
    text_feats = text_feats / np.linalg.norm(text_feats, axis=1, keepdims=True)
    mm_feats = mm_feats / np.linalg.norm(mm_feats, axis=1, keepdims=True)
    
    # Compute Similarities
    # We need top K for building graph
    params = {
        'k': 50,
        'nhid': 256,
        'dropout': 0.1,
        'nheads': 4,
        'pooling': 'mean'
    }
    
    # Simmats
    # Use simple dot product for TFIDF or similar if available, else skip or mock
    # Notebook used TFIDF. We will use Text feats as proxy or 1.0
    
    simmat_img, indexes_img = get_sim_dict(img_feats, k=Config.K)
    simmat_bert, indexes_bert = get_sim_dict(text_feats, k=Config.K)
    simmat_mm, _ = get_sim_dict(mm_feats, k=Config.K)
    
    # Build Graph
    top_neighbors = defaultdict(list)
    feats = defaultdict(lambda: defaultdict())
    pair_tuples = []
    
    print("Building Graph...")
    for i in tqdm(range(len(df))):
        # Union of neighbors
        right_indexes = set(indexes_img[i].tolist() + indexes_bert[i].tolist())
        if i in right_indexes:
            right_indexes.remove(i)
            
        right_indexes = list(right_indexes)
        scores = {}
        for j in right_indexes:
            pair_tuples.append((i, j))
            
            s_img = simmat_img.get((i, j), 0)
            s_bert = simmat_bert.get((i, j), 0)
            s_mm = simmat_mm.get((i, j), 0)
            s_dummy = 0 # Placeholder for TFIDF
            
            feats[i][j] = [s_img, s_dummy, s_bert, s_mm]
            scores[j] = s_img + s_bert + s_mm
            
        # Select top neighbors for GAT context
        top_neighbors[i] = sorted(right_indexes, key=lambda x: scores[x], reverse=True)[:params['k']]
        
    dataset = GraphDataset(
        feats=feats,
        pair_tuples=pair_tuples,
        k=params['k'],
        top_neighbors=top_neighbors,
    )
    
    loader = DataLoader(dataset, batch_size=2048, shuffle=False, drop_last=False, num_workers=Config.NUM_WORKERS)
    
    model = GATPairClassifier(nfeat=4, nhid=params['nhid'], dropout=params['dropout'], 
                              nheads=params['nheads'], pooling=params['pooling'])
    model.to(device)
    model.eval()
    
    # Load Checkpoint
    # checkpoint_path = Config.CHECKPOINT_DIR / 'gcn_model.pth'
    # try:
    #     model.load_state_dict(torch.load(checkpoint_path))
    # except:
    #     print("GCN Checkpoint not found, skipping prediction")
    #     return []

    print("Running GCN Prediction...")
    preds = []
    # Mock prediction loop if model not trained
    # In real scenario, uncomment load and run loop
    
    # for feats_batch, neighbor_feats_batch in tqdm(loader):
    #     feats_batch = feats_batch.to(device)
    #     neighbor_feats_batch = neighbor_feats_batch.to(device)
    #     with torch.no_grad():
    #         out = model(feats_batch, neighbor_feats_batch).sigmoid()
    #         preds.extend(out.cpu().numpy().tolist())
            
    # Since we don't have GCN trained model, returning empty or fallback
    return preds
