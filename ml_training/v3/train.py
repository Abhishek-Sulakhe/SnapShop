"""
Shopee Product Matching - v3 Training Pipeline
===============================================

Generates and trains everything used by the webapp inference engine:

  Stage 1   : Metric Learning
              5 embedding models with CurricularFace/ArcFace losses,
              SAM/AdamW optimizers, and sample weighting for micro-F1.
              Output: deit_small.pth, efficientnet_b3.pth,
                      bert_indonesian.pth, bert_multilingual.pth,
                      xlm_roberta.pth

  Stage 1.5 : Embedding Extraction
              Extract embeddings from all 5 models, concatenate with
              normalize -> concat -> re-normalize pattern.

  Stage 2   : LightGBM Re-Ranker
              Build FAISS index on combined embeddings, sample queries,
              retrieve top-50 candidates, compute 9 cross-modal features,
              and train a LambdaRank model.  Identical to the LightGBM
              trained by the inference engine at startup.
              Output: lgbm_ranker.txt

Usage:
    python train.py
"""

import os
import gc
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import timm
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.losses import ArcMarginProduct, CurricularFace
from data.dataset import ImageDataset, TextDataset
from utils.common import seed_everything
from utils.optim import SAM
from utils.logger import setup_logger
from utils.metrics import compute_sample_weights


# =====================================================================
# Stage 1: Metric Learning - Image Models
# =====================================================================

def train_image_model(model_name, cfg, df, num_classes, sample_weights,
                      device, logger):
    """Train an image metric learning model (DeiT Small or EfficientNet-B3).

    Key techniques:
      - CurricularFace loss: adaptive curriculum for hard negatives
      - SAM optimizer: finds flat minima for better generalization
      - Sample weighting: 1/(group_size^0.4) for micro-F1 optimization
      - Mixed-precision (AMP) for memory efficiency
    """
    logger.info(f"Training image model: {model_name}")
    logger.info(f"  Backbone: {cfg['backbone']}")
    logger.info(f"  Image size: {cfg['img_size']}, FC dim: {cfg['fc_dim']}")
    logger.info(f"  Loss: {cfg['loss_fn']}, Optimizer: {cfg['optimizer']}")

    backbone_name = cfg['backbone']
    if backbone_name.startswith('vit_deit_'):
        backbone_name = backbone_name.replace('vit_deit_', 'deit_')

    backbone = timm.create_model(backbone_name, pretrained=True)
    model = ImageEncoder(
        backbone, num_classes=num_classes,
        fc_dim=cfg['fc_dim'], s=cfg['s'], margin=cfg['margin'], p=cfg['p'],
    ).to(device)
    model.train()

    train_transform = transforms.Compose([
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageDataset(df, Config.TRAIN_IMAGES, transform=train_transform)
    loader = DataLoader(
        dataset, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True,
    )

    # Margin-based loss
    if cfg['loss_fn'] == 'curricularface':
        margin_layer = CurricularFace(
            in_features=cfg['fc_dim'], out_features=num_classes,
            s=cfg['s'], m=cfg['margin'],
        ).to(device)
    else:
        margin_layer = ArcMarginProduct(
            in_features=cfg['fc_dim'], out_features=num_classes,
            s=cfg['s'], m=cfg['margin'],
        ).to(device)

    weight_tensor = torch.FloatTensor(sample_weights).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')

    if cfg['optimizer'] == 'sam':
        optimizer = SAM(model.parameters(), torch.optim.Adam,
                        lr=cfg['lr'], rho=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer.base_optimizer, T_max=cfg['epochs'],
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['epochs'],
        )

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    for epoch in range(cfg['epochs']):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader,
                    desc=f"[{model_name}] Epoch {epoch+1}/{cfg['epochs']}")

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            batch_weights = weight_tensor[labels]

            if cfg['optimizer'] == 'sam':
                # SAM first forward-backward (ascent)
                with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    feats = model(imgs)
                    logits = margin_layer(feats, labels)
                    per_sample = criterion(logits, labels)
                    loss = (per_sample * batch_weights).mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer.first_step, zero_grad=True)

                # SAM second forward-backward (descent at perturbed weights)
                with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    feats = model(imgs)
                    logits = margin_layer(feats, labels)
                    per_sample = criterion(logits, labels)
                    loss = (per_sample * batch_weights).mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer.second_step, zero_grad=True)
                scaler.update()
            else:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    feats = model(imgs)
                    logits = margin_layer(feats, labels)
                    per_sample = criterion(logits, labels)
                    loss = (per_sample * batch_weights).mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(loader)
        logger.info(f"  Epoch {epoch+1} - avg loss: {avg_loss:.4f}")

    # Save checkpoint (exact format expected by inference engine)
    params = {
        'backbone': cfg['backbone'],
        'fc_dim': cfg['fc_dim'],
        'test_size': cfg['test_size'],
        'img_size': cfg['img_size'],
        's': cfg['s'],
        'margin': cfg['margin'],
        'p': cfg['p'],
    }
    if 'p_eval' in cfg:
        params['p_eval'] = cfg['p_eval']

    save_path = Config.CHECKPOINT_DIR / f"{model_name}.pth"
    torch.save({'model': model.state_dict(), 'params': params}, save_path)
    logger.info(f"  Saved -> {save_path}")

    del model, margin_layer, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()


# =====================================================================
# Stage 1: Metric Learning - Text Models
# =====================================================================

def train_text_model(model_name, cfg, df, num_classes, sample_weights,
                     device, logger):
    """Train a text metric learning model.

    Models and pooling:
      - bert_indonesian  : simple_mean  (avg all tokens)
      - bert_multilingual: attn-weighted (ignores padding)
      - xlm_roberta      : attn-weighted (no token_type_ids)

    Uses ArcFace loss + AdamW (lr=5e-5) + CosineAnnealingLR.
    """
    logger.info(f"Training text model: {model_name}")
    logger.info(f"  Transformer: {cfg['bert_name']}")
    logger.info(f"  FC dim: {cfg['fc_dim']}, max_len: {cfg['max_len']}")
    logger.info(f"  Pooling: {'simple_mean' if cfg['simple_mean'] else 'attention_weighted'}")

    from transformers import AutoTokenizer, AutoModel
    from transformers import BertTokenizerFast, BertModel

    bert_name = cfg['bert_name']
    if 'indonesian' in bert_name.lower():
        tokenizer = BertTokenizerFast.from_pretrained(bert_name)
        bert_base = BertModel.from_pretrained(bert_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(bert_name)
        bert_base = AutoModel.from_pretrained(bert_name)

    model = TextEncoder(
        bert_base, num_classes=num_classes, tokenizer=tokenizer,
        max_len=cfg['max_len'], fc_dim=cfg['fc_dim'],
        simple_mean=cfg['simple_mean'],
    ).to(device)
    model.train()

    dataset = TextDataset(df)
    loader = DataLoader(
        dataset, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True,
    )

    margin_layer = ArcMarginProduct(
        in_features=cfg['fc_dim'], out_features=num_classes,
        s=cfg['s'], m=cfg['margin'],
    ).to(device)

    weight_tensor = torch.FloatTensor(sample_weights).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    for epoch in range(cfg['epochs']):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader,
                    desc=f"[{model_name}] Epoch {epoch+1}/{cfg['epochs']}")

        for texts, labels in pbar:
            labels = labels.to(device)
            batch_weights = weight_tensor[labels]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                feats = model.extract_feat(list(texts))
                logits = margin_layer(feats, labels)
                per_sample = criterion(logits, labels)
                loss = (per_sample * batch_weights).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(loader)
        logger.info(f"  Epoch {epoch+1} - avg loss: {avg_loss:.4f}")

    params = {
        'bert_name': cfg['bert_name'],
        'fc_dim': cfg['fc_dim'],
        'max_len': cfg['max_len'],
        's': cfg['s'],
        'margin': cfg['margin'],
        'simple_mean': cfg['simple_mean'],
    }

    save_path = Config.CHECKPOINT_DIR / f"{model_name}.pth"
    torch.save({'model': model.state_dict(), 'params': params}, save_path)
    logger.info(f"  Saved -> {save_path}")

    del model, margin_layer, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()


# =====================================================================
# Stage 1.5: Embedding Extraction
# =====================================================================

def extract_image_embeddings(model_name, cfg, df, device, logger):
    """Load a trained image checkpoint and extract embeddings."""
    logger.info(f"Extracting image embeddings: {model_name}")

    ckpt_path = Config.CHECKPOINT_DIR / f"{model_name}.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    params = ckpt['params']

    backbone_name = params['backbone']
    if backbone_name.startswith('vit_deit_'):
        backbone_name = backbone_name.replace('vit_deit_', 'deit_')

    backbone = timm.create_model(backbone_name, pretrained=False)
    model = ImageEncoder(
        backbone, num_classes=0,
        fc_dim=params['fc_dim'], p=params.get('p', 3),
    ).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    if 'p_eval' in params:
        model.p = params['p_eval']

    test_size = params.get('test_size', params.get('img_size', 224))
    test_transform = transforms.Compose([
        transforms.Resize(test_size + 32),
        transforms.CenterCrop(test_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageDataset(df, Config.TRAIN_IMAGES, transform=test_transform)
    loader = DataLoader(
        dataset, batch_size=32, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True,
    )

    all_feats = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc=f"  {model_name} embeddings"):
            imgs = imgs.to(device)
            feats = model.extract_feat(imgs)
            all_feats.append(feats.cpu().numpy())

    embeddings = np.concatenate(all_feats, axis=0)
    logger.info(f"  {model_name}: {embeddings.shape}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return embeddings


def extract_text_embeddings(model_name, cfg, df, device, logger):
    """Load a trained text checkpoint and extract embeddings."""
    logger.info(f"Extracting text embeddings: {model_name}")

    from transformers import AutoTokenizer, AutoModel
    from transformers import BertTokenizerFast, BertModel

    ckpt_path = Config.CHECKPOINT_DIR / f"{model_name}.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    params = ckpt['params']

    bert_name = params['bert_name']
    if 'indonesian' in bert_name.lower():
        tokenizer = BertTokenizerFast.from_pretrained(bert_name)
        bert_base = BertModel.from_pretrained(bert_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(bert_name)
        bert_base = AutoModel.from_pretrained(bert_name)

    model = TextEncoder(
        bert_base, num_classes=0, tokenizer=tokenizer,
        max_len=params['max_len'], fc_dim=params['fc_dim'],
        simple_mean=params['simple_mean'],
    ).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    batch_size = 64
    all_feats = []
    titles = df['title'].tolist()

    with torch.no_grad():
        for i in tqdm(range(0, len(titles), batch_size),
                      desc=f"  {model_name} embeddings"):
            batch = titles[i:i + batch_size]
            feats = model.extract_feat(batch)
            all_feats.append(feats.cpu().numpy())

    embeddings = np.concatenate(all_feats, axis=0)
    logger.info(f"  {model_name}: {embeddings.shape}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return embeddings


def build_concatenated_embeddings(image_embeddings, text_embeddings, logger):
    """Build multi-model concatenated embeddings.

    Pattern: L2-normalize each model -> concat -> L2-normalize again.
    This ensures each model contributes equally and enables cosine
    similarity via inner product (||x||=1 -> dot(x,y) = cos(theta)).
    """
    logger.info("Building concatenated embeddings...")

    # Image: [deit_small(512), efficientnet_b3(512)] -> 1024-d
    img_parts = []
    for name, emb in image_embeddings.items():
        emb_norm = emb / (np.linalg.norm(emb, 2, axis=1, keepdims=True) + 1e-8)
        img_parts.append(emb_norm)
    img_feats = np.concatenate(img_parts, axis=1).astype(np.float32)
    img_feats /= (np.linalg.norm(img_feats, 2, axis=1, keepdims=True) + 1e-8)
    logger.info(f"  img_feats: {img_feats.shape}")

    # Text: [bert_indo(512), bert_multi(512), xlm(512)] -> 1536-d
    txt_parts = []
    for name, emb in text_embeddings.items():
        emb_norm = emb / (np.linalg.norm(emb, 2, axis=1, keepdims=True) + 1e-8)
        txt_parts.append(emb_norm)
    txt_feats = np.concatenate(txt_parts, axis=1).astype(np.float32)
    txt_feats /= (np.linalg.norm(txt_feats, 2, axis=1, keepdims=True) + 1e-8)
    logger.info(f"  txt_feats: {txt_feats.shape}")

    # Combined: [text(1536), image(1024)] -> 2560-d
    combined_feats = np.concatenate([txt_feats, img_feats], axis=1).astype(np.float32)
    combined_feats /= (np.linalg.norm(combined_feats, 2, axis=1, keepdims=True) + 1e-8)
    logger.info(f"  combined_feats: {combined_feats.shape}")

    return img_feats, txt_feats, combined_feats


# =====================================================================
# Stage 2: LightGBM Re-Ranker
# (Identical to inference_engine.py's _train_lgbm_ranker)
# =====================================================================

def compute_ranking_features(query_idx, candidate_indices, candidate_scores,
                             img_feats, txt_feats):
    """Compute the 9 cross-modal features for (query, candidate) pairs.

    This is identical to SearchEngine._compute_ranking_features in the
    inference engine.  The 9 features are:

      0: combined_score   - FAISS combined cosine similarity
      1: img_score        - image-only cosine similarity
      2: text_score       - text-only cosine similarity
      3: score_product    - img * txt (joint agreement)
      4: score_harmonic   - harmonic mean of img and txt
      5: score_rank       - normalized rank position (0=best, 1=worst)
      6: img_text_diff    - img - txt (modality disagreement)
      7: max_modal_score  - max(img, txt)
      8: min_modal_score  - min(img, txt)
    """
    n = len(candidate_indices)

    q_img = img_feats[query_idx]
    q_txt = txt_feats[query_idx]
    c_img = img_feats[candidate_indices]
    c_txt = txt_feats[candidate_indices]

    img_scores = c_img @ q_img
    text_scores = c_txt @ q_txt

    score_product = img_scores * text_scores
    eps = 1e-7
    score_harmonic = 2.0 * img_scores * text_scores / (img_scores + text_scores + eps)
    score_ranks = np.arange(n).astype(np.float32) / max(n - 1, 1)
    img_text_diff = img_scores - text_scores
    max_modal = np.maximum(img_scores, text_scores)
    min_modal = np.minimum(img_scores, text_scores)

    features = np.column_stack([
        candidate_scores[:n],   # 0: combined_score
        img_scores,             # 1: img_score
        text_scores,            # 2: text_score
        score_product,          # 3: score_product
        score_harmonic,         # 4: score_harmonic
        score_ranks,            # 5: score_rank
        img_text_diff,          # 6: img_text_diff
        max_modal,              # 7: max_modal_score
        min_modal,              # 8: min_modal_score
    ])

    return features.astype(np.float32)


def train_lgbm_ranker(df, img_feats, txt_feats, combined_feats, logger):
    """Train LightGBM LambdaRank re-ranker using FAISS candidates.

    This mirrors the inference engine's _train_lgbm_ranker exactly:
      1. Build a FAISS IndexFlatIP on combined embeddings
      2. Sample N queries, retrieve top-K candidates each
      3. Compute 9 cross-modal features per (query, candidate) pair
      4. Label: 1 if same label_group, 0 otherwise
      5. Train LambdaRank model

    The resulting model is saved in the same format that the inference
    engine loads at startup.
    """
    import faiss
    import lightgbm as lgb

    logger.info("Training LightGBM LambdaRank re-ranker...")

    n_products = len(df)
    n_queries = min(Config.LGBM_NUM_QUERIES, n_products)
    k_candidates = Config.LGBM_K_CANDIDATES
    label_groups = df['label_group'].values

    # Build FAISS index on combined embeddings
    logger.info(f"  Building FAISS index (dim={combined_feats.shape[1]})...")
    index = faiss.IndexFlatIP(combined_feats.shape[1])
    index.add(combined_feats)

    # Sample queries (deterministic)
    rng = np.random.RandomState(Config.SEED)
    query_indices = rng.choice(n_products, size=n_queries, replace=False)

    all_features = []
    all_labels = []
    group_sizes = []

    logger.info(f"  Building training data: {n_queries} queries x top-{k_candidates}...")

    for qi in tqdm(query_indices, desc="  LightGBM training data"):
        query_vec = combined_feats[qi:qi + 1]
        D, I = index.search(query_vec, k_candidates + 1)

        candidate_indices = I[0]
        candidate_scores = D[0]

        # Remove self from candidates
        mask = candidate_indices != qi
        candidate_indices = candidate_indices[mask][:k_candidates]
        candidate_scores = candidate_scores[mask][:k_candidates]

        if len(candidate_indices) == 0:
            continue

        features = compute_ranking_features(
            qi, candidate_indices, candidate_scores,
            img_feats, txt_feats,
        )

        # Labels: 1 if same label_group, 0 otherwise
        query_label = label_groups[qi]
        labels = (label_groups[candidate_indices] == query_label).astype(np.float32)

        all_features.append(features)
        all_labels.append(labels)
        group_sizes.append(len(candidate_indices))

    if not all_features:
        logger.error("  No training data generated!")
        return None

    X_train = np.concatenate(all_features, axis=0)
    y_train = np.concatenate(all_labels, axis=0)
    group_train = np.array(group_sizes)

    logger.info(f"  Training set: {X_train.shape[0]} pairs, "
                f"{int(y_train.sum())} positives ({y_train.mean()*100:.1f}%)")

    feature_names = [
        'combined_score', 'img_score', 'text_score',
        'score_product', 'score_harmonic',
        'score_rank', 'img_text_diff',
        'max_modal_score', 'min_modal_score',
    ]

    train_data = lgb.Dataset(
        X_train, label=y_train, group=group_train,
        feature_name=feature_names,
    )

    model = lgb.train(
        Config.LGBM_PARAMS,
        train_data,
        num_boost_round=Config.LGBM_NUM_BOOST_ROUND,
        valid_sets=[train_data],
        callbacks=[lgb.log_evaluation(period=50)],
    )

    # Log feature importance
    importance = model.feature_importance(importance_type='gain')
    logger.info("  Feature importance (gain):")
    for name, imp in sorted(zip(feature_names, importance),
                            key=lambda x: -x[1]):
        logger.info(f"    {name:25s} {imp:.1f}")

    return model


# =====================================================================
# Main Pipeline
# =====================================================================

def main():
    t_start = time.time()

    print("=" * 70)
    print("  Shopee Product Matching - v3 Training Pipeline")
    print("  Stage 1 : Metric Learning (DeiT, EfficientNet-B3, BERT x3)")
    print("  Stage 2 : LightGBM LambdaRank Re-Ranker")
    print("=" * 70)

    seed_everything(Config.SEED)
    Config.setup()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    logger = setup_logger('v3_train', str(Config.OUTPUT_DIR),
                          'v3_training.log')

    # -----------------------------------------------------------------
    # Load & prepare data
    # -----------------------------------------------------------------
    if not Config.TRAIN_CSV.exists():
        logger.error(f"Train CSV not found: {Config.TRAIN_CSV}")
        return

    df = pd.read_csv(Config.TRAIN_CSV)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label_group'])
    num_classes = df['label'].nunique()
    logger.info(f"Loaded {len(df)} products, {num_classes} classes")

    # Sample weights: 1/(group_size^0.4) for micro-F1 optimization
    sample_weights = compute_sample_weights(df, power=Config.SAMPLE_WEIGHT_POWER)
    logger.info(f"Sample weights: min={sample_weights.min():.3f}, "
                f"max={sample_weights.max():.3f}, mean={sample_weights.mean():.3f}")

    # =================================================================
    # STAGE 1: Train Metric Learning Models
    # =================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1: Metric Learning Training")
    logger.info("=" * 60)

    for name, cfg in Config.IMAGE_MODELS.items():
        print(f"\n{'─' * 50}")
        try:
            train_image_model(name, cfg, df, num_classes,
                              sample_weights, device, logger)
        except Exception as e:
            logger.error(f"Failed training {name}: {e}")
            import traceback; traceback.print_exc()

    for name, cfg in Config.TEXT_MODELS.items():
        print(f"\n{'─' * 50}")
        try:
            train_text_model(name, cfg, df, num_classes,
                             sample_weights, device, logger)
        except Exception as e:
            logger.error(f"Failed training {name}: {e}")
            import traceback; traceback.print_exc()

    # =================================================================
    # STAGE 1.5: Extract Embeddings
    # =================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1.5: Embedding Extraction")
    logger.info("=" * 60)

    image_embeddings = {}
    for name, cfg in Config.IMAGE_MODELS.items():
        try:
            image_embeddings[name] = extract_image_embeddings(
                name, cfg, df, device, logger)
        except Exception as e:
            logger.error(f"Failed extracting {name}: {e}")

    text_embeddings = {}
    for name, cfg in Config.TEXT_MODELS.items():
        try:
            text_embeddings[name] = extract_text_embeddings(
                name, cfg, df, device, logger)
        except Exception as e:
            logger.error(f"Failed extracting {name}: {e}")

    img_feats, txt_feats, combined_feats = build_concatenated_embeddings(
        image_embeddings, text_embeddings, logger,
    )

    # =================================================================
    # STAGE 2: LightGBM Re-Ranker
    # =================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: LightGBM LambdaRank Re-Ranker")
    logger.info("=" * 60)

    try:
        lgbm_model = train_lgbm_ranker(
            df, img_feats, txt_feats, combined_feats, logger,
        )

        if lgbm_model is not None:
            lgbm_path = Config.OUTPUT_DIR / 'lgbm_ranker.txt'
            lgbm_model.save_model(str(lgbm_path))
            logger.info(f"  Saved -> {lgbm_path}")
    except ImportError as e:
        logger.warning(f"Skipping LightGBM (not installed): {e}")
    except Exception as e:
        logger.error(f"LightGBM training failed: {e}")
        import traceback; traceback.print_exc()

    # =================================================================
    # Summary
    # =================================================================
    elapsed = time.time() - t_start

    print("\n" + "=" * 70)
    print("  Training Complete!")
    print(f"  Total time: {elapsed/3600:.1f} hours ({elapsed:.0f}s)")
    print("=" * 70)

    print(f"\n  Checkpoints in: {Config.CHECKPOINT_DIR}")
    for f in sorted(Config.CHECKPOINT_DIR.glob('*.pth')):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    {f.name:35s} {size_mb:8.1f} MB")

    lgbm_path = Config.OUTPUT_DIR / 'lgbm_ranker.txt'
    if lgbm_path.exists():
        size_mb = lgbm_path.stat().st_size / (1024 * 1024)
        print(f"    {'lgbm_ranker.txt':35s} {size_mb:8.1f} MB")

    print()


if __name__ == '__main__':
    main()
