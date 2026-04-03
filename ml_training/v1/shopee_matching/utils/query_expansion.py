import numpy as np

def query_expansion(feats, sims, topk_idx, alpha=0.5, k=2):
    weights = np.expand_dims(sims[:, :k] ** alpha, axis=-1).astype(np.float32)
    feats = (feats[topk_idx[:, :k]] * weights).sum(axis=1)
    return feats
