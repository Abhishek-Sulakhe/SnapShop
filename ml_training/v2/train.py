"""
Shopee Product Matching — v2 Training Pipeline
Trains all 6 models used in the webapp inference engine:
  Image:      DeiT Small, EfficientNet-B3
  Text:       BERT Indonesian, BERT Multilingual, XLM-RoBERTa
  Multimodal: DeiT + BERT (multimodal fusion)
"""

import os
import sys
import torch
import torch.nn as nn
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
from data.dataset import ImageDataset, TextDataset, MultiModalDataset
from utils.common import seed_everything
from utils.optim import SAM
from utils.logger import setup_logger


# Image Model Training (DeiT Small / EfficientNet-B3)

def train_image_model(model_name, cfg, df, num_classes, device, logger):
    logger.info(f"Training image model: {model_name}")
    logger.info(f"  Backbone: {cfg['backbone']}")
    logger.info(f"  Image size: {cfg['img_size']}, FC dim: {cfg['fc_dim']}")

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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageDataset(df, Config.TRAIN_IMAGES, transform=train_transform)
    loader = DataLoader(
        dataset, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True,
    )

    # Loss
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

    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if cfg['optimizer'] == 'sam':
        optimizer = SAM(model.parameters(), torch.optim.Adam, lr=cfg['lr'], rho=0.05)
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
        pbar = tqdm(loader, desc=f"[{model_name}] Epoch {epoch+1}/{cfg['epochs']}")

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            if cfg['optimizer'] == 'sam':
                # SAM first step
                with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    feats = model(imgs)
                    logits = margin_layer(feats, labels)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer.first_step, zero_grad=True)

                # SAM second step
                with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    feats = model(imgs)
                    logits = margin_layer(feats, labels)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer.second_step, zero_grad=True)
                scaler.update()
            else:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    feats = model(imgs)
                    logits = margin_layer(feats, labels)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(loader)
        logger.info(f"  Epoch {epoch+1} — avg loss: {avg_loss:.4f}")

    # Save checkpoint in the format the inference engine expects
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
    logger.info(f"  Saved → {save_path.relative_to(Config.PROJECT_ROOT)}")


# Text Model Training (BERT Indonesian / Multilingual / XLM-RoBERTa)

def train_text_model(model_name, cfg, df, num_classes, device, logger):
    logger.info(f"Training text model: {model_name}")
    logger.info(f"  Transformer: {cfg['bert_name']}")
    logger.info(f"  FC dim: {cfg['fc_dim']}, max_len: {cfg['max_len']}")

    from transformers import AutoTokenizer, AutoModel, BertTokenizerFast, BertModel

    bert_name = cfg['bert_name']

    # Load tokenizer + base model
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

    # Loss (always ArcFace for text models)
    margin_layer = ArcMarginProduct(
        in_features=cfg['fc_dim'], out_features=num_classes,
        s=cfg['s'], m=cfg['margin'],
    ).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    for epoch in range(cfg['epochs']):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"[{model_name}] Epoch {epoch+1}/{cfg['epochs']}")

        for texts, labels in pbar:
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                feats = model.extract_feat(list(texts))
                logits = margin_layer(feats, labels)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(loader)
        logger.info(f"  Epoch {epoch+1} — avg loss: {avg_loss:.4f}")

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
    logger.info(f"  Saved → {save_path.relative_to(Config.PROJECT_ROOT)}")


# Multimodal Model Training (DeiT + BERT fusion)

def train_multimodal_model(model_name, cfg, df, num_classes, device, logger):
    from transformers import BertTokenizerFast, BertModel
    from models.image_encoder import ImageEncoder  # reuse gem pooling logic

    logger.info(f"Training multimodal model: {model_name}")
    logger.info(f"  Backbone: {cfg['backbone']} + {cfg['bert_name']}")

    # Build multimodal net inline (same arch as inference engine's MultiModalNet)
    class MultiModalNet(nn.Module):
        def __init__(self, backbone, bert_model, tokenizer,
                     fc_dim=512, max_len=128, p=3):
            super().__init__()
            from models.image_encoder import gem
            self.gem = gem
            self.backbone = backbone
            self.backbone.reset_classifier(num_classes=0)
            self.bert_model = bert_model
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.p = p
            self.fc = nn.Linear(
                self.bert_model.config.hidden_size + self.backbone.num_features, fc_dim,
            )
            self.bn = nn.BatchNorm1d(fc_dim)
            nn.init.xavier_normal_(self.fc.weight)
            nn.init.constant_(self.fc.bias, 0)
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)

        def forward(self, img, title):
            batch_size = img.shape[0]
            device = img.device
            img_feat = self.backbone.forward_features(img)
            if len(img_feat.shape) == 3:
                img_feat = (img_feat[:, 0] + img_feat[:, 1]) / 2
            else:
                img_feat = self.gem(img_feat, p=self.p).view(batch_size, -1)

            tok = self.tokenizer(
                title, truncation=True, padding=True, max_length=self.max_len,
            )
            input_ids = torch.LongTensor(tok['input_ids']).to(device)
            token_type_ids = torch.LongTensor(tok['token_type_ids']).to(device)
            attention_mask = torch.LongTensor(tok['attention_mask']).to(device)

            txt_out = self.bert_model(
                input_ids=input_ids, token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            txt_feat = txt_out.last_hidden_state.mean(dim=1)

            x = torch.cat([img_feat, txt_feat], dim=1)
            x = self.fc(x)
            x = self.bn(x)
            return x

    # Build model
    backbone_name = cfg['backbone']
    if backbone_name.startswith('vit_deit_'):
        backbone_name = backbone_name.replace('vit_deit_', 'deit_')

    backbone = timm.create_model(backbone_name, pretrained=True)
    tokenizer = BertTokenizerFast.from_pretrained(cfg['bert_name'])
    bert_base = BertModel.from_pretrained(cfg['bert_name'])

    model = MultiModalNet(
        backbone, bert_base, tokenizer,
        fc_dim=cfg['fc_dim'], max_len=cfg['max_len'], p=cfg['p'],
    ).to(device)
    model.train()

    train_transform = transforms.Compose([
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = MultiModalDataset(
        df, Config.TRAIN_IMAGES, tokenizer,
        transform=train_transform, max_len=cfg['max_len'],
    )
    loader = DataLoader(
        dataset, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True,
    )

    margin_layer = ArcMarginProduct(
        in_features=cfg['fc_dim'], out_features=num_classes,
        s=cfg['s'], m=cfg['margin'],
    ).to(device)
    criterion = nn.CrossEntropyLoss()

    if cfg['optimizer'] == 'sam':
        optimizer = SAM(model.parameters(), torch.optim.Adam, lr=cfg['lr'], rho=0.05)
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
        pbar = tqdm(loader, desc=f"[{model_name}] Epoch {epoch+1}/{cfg['epochs']}")

        for imgs, input_ids, attention_mask, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            titles_batch = [
                dataset.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                for i in range(imgs.size(0))
            ]

            if cfg['optimizer'] == 'sam':
                with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    feats = model(imgs, titles_batch)
                    logits = margin_layer(feats, labels)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer.first_step, zero_grad=True)

                with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    feats = model(imgs, titles_batch)
                    logits = margin_layer(feats, labels)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer.second_step, zero_grad=True)
                scaler.update()
            else:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    feats = model(imgs, titles_batch)
                    logits = margin_layer(feats, labels)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(loader)
        logger.info(f"  Epoch {epoch+1} — avg loss: {avg_loss:.4f}")

    params = {
        'backbone': cfg['backbone'],
        'bert_name': cfg['bert_name'],
        'fc_dim': cfg['fc_dim'],
        'max_len': cfg['max_len'],
        'test_size': cfg['test_size'],
        'img_size': cfg['img_size'],
        'p': cfg['p'],
        's': cfg['s'],
        'margin': cfg['margin'],
    }

    save_path = Config.CHECKPOINT_DIR / f"{model_name}.pth"
    torch.save({'model': model.state_dict(), 'params': params}, save_path)
    logger.info(f"  Saved → {save_path.relative_to(Config.PROJECT_ROOT)}")


# Main Orchestrator

def main():
    print("=" * 60)
    print("  Shopee Product Matching — v2 Training Pipeline")
    print("  Models: DeiT Small, EfficientNet-B3, BERT Indo,")
    print("          BERT Multilingual, XLM-RoBERTa, DeiT+BERT")
    print("=" * 60)

    seed_everything(Config.SEED)
    Config.setup()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    logger = setup_logger('v2_train', str(Config.OUTPUT_DIR), 'v2_training.log')

    # Load data
    if not Config.TRAIN_CSV.exists():
        logger.error(f"Train CSV not found: {Config.TRAIN_CSV}")
        return

    df = pd.read_csv(Config.TRAIN_CSV)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label_group'])
    num_classes = df['label'].nunique()
    logger.info(f"Loaded {len(df)} products, {num_classes} classes")

    # ── Stage 1: Image Models ──
    for name, cfg in Config.IMAGE_MODELS.items():
        print(f"\n{'─' * 50}")
        try:
            train_image_model(name, cfg, df, num_classes, device, logger)
        except Exception as e:
            logger.error(f"Failed training {name}: {e}")

    # ── Stage 2: Text Models ──
    for name, cfg in Config.TEXT_MODELS.items():
        print(f"\n{'─' * 50}")
        try:
            train_text_model(name, cfg, df, num_classes, device, logger)
        except Exception as e:
            logger.error(f"Failed training {name}: {e}")

    # ── Stage 3: Multimodal Models ──
    for name, cfg in Config.MULTIMODAL_MODELS.items():
        print(f"\n{'─' * 50}")
        try:
            train_multimodal_model(name, cfg, df, num_classes, device, logger)
        except Exception as e:
            logger.error(f"Failed training {name}: {e}")

    print("\n" + "=" * 60)
    print("  Training Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
