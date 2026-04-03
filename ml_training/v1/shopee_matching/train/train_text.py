import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from transformers import BertTokenizerFast, BertConfig, BertModel
from sklearn.preprocessing import LabelEncoder
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from shopee_matching.config import Config
from shopee_matching.utils.common import seed_everything
from shopee_matching.models.text_encoders import BertNet
from shopee_matching.utils.losses import ArcMarginProduct
from shopee_matching.utils.optim import SAM
from shopee_matching.data.dataset import BertDataset
from shopee_matching.utils.logger import setup_logger

def train_text_model():
    seed_everything(Config.SEED)
    
    logger = setup_logger("TrainText", Config.OUTPUT_DIR, "train_text.log")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Config
    model_name = 'bert-base-multilingual-cased' 
    batch_size = 16
    epochs = 10
    out_dim = Config.FC_DIM
    
    if not os.path.exists(Config.TRAIN_CSV):
        logger.error(f"Train CSV not found at {Config.TRAIN_CSV}")
        return

    df = pd.read_csv(Config.TRAIN_CSV)
    
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label_group'])
    num_classes = df['label'].nunique()
    logger.info(f"Found {num_classes} classes.")
    
    # Tokenizer
    try:
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
    except:
        # Fallback to local if downloaded
        tokenizer = BertTokenizerFast.from_pretrained(Config.BERT_MULTILINGUAL_DIR)

    # Data
    dataset = BertDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True)
    
    # Model
    bert_model = BertModel.from_pretrained(model_name)
    model = BertNet(bert_model, num_classes=num_classes, tokenizer=tokenizer, fc_dim=out_dim)
    model = model.to(device)
    model.train()
    
    # Metric Layer (ArcFace)
    margin_layer = ArcMarginProduct(in_features=out_dim, out_features=num_classes).to(device)
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer - Using AdamW for BERT is standard
    start_lr = 5e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    scaler = torch.cuda.amp.GradScaler()
    
    logger.info("Starting Text Model Training...")
    os.makedirs(Config.CHECKPOINT_SAVE_DIR, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        avg_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (text, label) in enumerate(pbar):
            label = label.to(device)
            # Text is a tuple/list from batch, tokenizer handles it inside model or here
            # Our BertNet handles list of strings
            
            optimizer.zero_grad()
            
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    feats = model.extract_feat(text)
                    output = margin_layer(feats, label)
                    loss = criterion(output, label)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                feats = model.extract_feat(text)
                output = margin_layer(feats, label)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                
            avg_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        scheduler.step()
        epoch_loss = avg_loss/len(dataloader)
        logger.info(f"Epoch {epoch+1} done. Avg Loss: {epoch_loss:.4f}")
        
        save_path = Config.CHECKPOINT_SAVE_DIR / f'bert_epoch_{epoch+1}.pth'
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'params': {
                'max_len': model.max_len,
                'fc_dim': out_dim,
                'model_name': model_name
            }
        }, save_path)
        logger.info(f"Saved checkpoint to {save_path}")

if __name__ == '__main__':
    train_text_model()
