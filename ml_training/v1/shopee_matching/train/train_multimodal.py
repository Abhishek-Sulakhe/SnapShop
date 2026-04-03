import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import timm
from transformers import BertTokenizerFast, BertConfig, BertModel
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from shopee_matching.config import Config
from shopee_matching.utils.common import seed_everything
from shopee_matching.models.multimodal import MultiModalNet
from shopee_matching.utils.losses import ArcMarginProduct
from shopee_matching.utils.optim import SAM
from shopee_matching.data.dataset import MultiModalDataset
from shopee_matching.utils.logger import setup_logger

def train_multimodal_model():
    seed_everything(Config.SEED)
    
    logger = setup_logger("TrainMultiModal", Config.OUTPUT_DIR, "train_multimodal.log")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Config
    img_backbone_name = 'tf_efficientnet_b0_ns'
    text_model_name = 'bert-base-multilingual-cased'
    batch_size = 8 # Smaller batch size for multimodal due to memory
    epochs = 6
    out_dim = Config.FC_DIM
    
    if not os.path.exists(Config.TRAIN_CSV):
        logger.error(f"Train CSV not found at {Config.TRAIN_CSV}")
        return

    df = pd.read_csv(Config.TRAIN_CSV)
    
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label_group'])
    num_classes = df['label'].nunique()
    logger.info(f"Found {num_classes} classes.")
    
    # Text Models
    try:
        bert_config = BertConfig.from_pretrained(text_model_name)
        bert_model = BertModel.from_pretrained(text_model_name, config=bert_config)
        tokenizer = BertTokenizerFast.from_pretrained(text_model_name)
    except:
         # Fallback
        tokenizer = BertTokenizerFast.from_pretrained(Config.BERT_MULTILINGUAL_DIR)
        bert_model = BertModel.from_pretrained(Config.BERT_MULTILINGUAL_DIR)
        
    # Image Backbone
    backbone = timm.create_model(img_backbone_name, pretrained=True)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset
    dataset = MultiModalDataset(df, Config.TRAIN_IMAGES, tokenizer, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True)
    
    # Model
    model = MultiModalNet(backbone, bert_model, num_classes=num_classes, tokenizer=tokenizer, fc_dim=out_dim)
    model = model.to(device)
    model.train()
    
    # Metric Layer
    margin_layer = ArcMarginProduct(in_features=out_dim, out_features=num_classes).to(device) # Can use CurricularFace too
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = SAM(model.parameters(), torch.optim.Adam, lr=1e-3, rho=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=epochs)
    
    scaler = torch.cuda.amp.GradScaler()
    
    logger.info("Starting Multimodal Training...")
    os.makedirs(Config.CHECKPOINT_SAVE_DIR, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        avg_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (img, input_ids, attention_mask, label) in enumerate(pbar):
            img = img.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)
            
            # First Step (SAM)
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    feats = model(img, input_ids, attention_mask)
                    output = margin_layer(feats, label)
                    loss = criterion(output, label)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer.first_step, zero_grad=True)
                
                # Second Step
                with torch.cuda.amp.autocast():
                    feats = model(img, input_ids, attention_mask)
                    output = margin_layer(feats, label)
                    loss = criterion(output, label)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer.second_step, zero_grad=True)
                scaler.update()
            else:
                feats = model(img, input_ids, attention_mask)
                output = margin_layer(feats, label)
                loss = criterion(output, label)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                feats = model(img, input_ids, attention_mask)
                output = margin_layer(feats, label)
                loss = criterion(output, label)
                loss.backward()
                optimizer.second_step(zero_grad=True)
                
            avg_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        scheduler.step()
        epoch_loss = avg_loss/len(dataloader)
        logger.info(f"Epoch {epoch+1} done. Avg Loss: {epoch_loss:.4f}")
        
        save_path = Config.CHECKPOINT_SAVE_DIR / f'multimodal_epoch_{epoch+1}.pth'
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'num_classes': num_classes,
            'fc_dim': out_dim
        }, save_path)

if __name__ == '__main__':
    train_multimodal_model()
