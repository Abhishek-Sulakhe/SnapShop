import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import timm
from pathlib import Path
import os
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
import sys

# Add project root to path to ensure modules can be imported
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from shopee_matching.config import Config
from shopee_matching.utils.common import seed_everything
from shopee_matching.models.image_encoders import ShopeeNet
from shopee_matching.utils.losses import CurricularFace
from shopee_matching.utils.optim import SAM
from shopee_matching.data.dataset import ShopeeDataset
from shopee_matching.utils.logger import setup_logger

def train_image_model():
    seed_everything(Config.SEED)
    
    # Setup Logger
    logger = setup_logger("TrainImage", Config.OUTPUT_DIR, "train_image.log")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Configuration
    model_name = 'tf_efficientnet_b0_ns' 
    out_dim = Config.FC_DIM
    batch_size = 16
    epochs = 8
    
    # Check for training data
    if not os.path.exists(Config.TRAIN_CSV):
        logger.error(f"Train CSV not found at {Config.TRAIN_CSV}. Please ensure data is placed correctly.")
        return

    df = pd.read_csv(Config.TRAIN_CSV)
    
    # Label Encoding
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label_group'])
    num_classes = df['label'].nunique()
    logger.info(f"Found {num_classes} classes in training data.")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset & Loader
    dataset = ShopeeDataset(df, Config.TRAIN_IMAGES, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True)
    
    # Model
    backbone = timm.create_model(model_name, pretrained=True)
    model = ShopeeNet(backbone, num_classes=num_classes, fc_dim=out_dim)
    model = model.to(device)
    model.train()
    
    # Metric Layer (CurricularFace)
    margin_layer = CurricularFace(in_features=out_dim, out_features=num_classes).to(device)
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    base_optimizer = torch.optim.Adam
    optimizer = SAM(model.parameters(), base_optimizer, lr=1e-3, rho=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=epochs)
    
    # Training Loop
    logger.info(f"Starting training for {epochs} epochs...")
    
    os.makedirs(Config.CHECKPOINT_SAVE_DIR, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        avg_loss = 0
        scaler = torch.cuda.amp.GradScaler() # Mixed precision for efficiency
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (img, label) in enumerate(pbar):
            img = img.to(device)
            label = label.to(device)
            
            # First Step (SAM)
            # Use AutoCast if on GPU
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    feats = model(img, label) # embeddings
                    output = margin_layer(feats, label) # logits
                    loss = criterion(output, label)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer.first_step, zero_grad=True)
                
                # Second Step (SAM)
                with torch.cuda.amp.autocast():
                    feats = model(img, label)
                    output = margin_layer(feats, label)
                    loss = criterion(output, label)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer.second_step, zero_grad=True)
                scaler.update()
            else:
                # CPU Fallback (no mixed precision/scaler usually)
                feats = model(img, label)
                output = margin_layer(feats, label)
                loss = criterion(output, label)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                feats = model(img, label)
                output = margin_layer(feats, label)
                loss = criterion(output, label)
                loss.backward()
                optimizer.second_step(zero_grad=True)

            avg_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        scheduler.step()
        epoch_loss = avg_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} done. Avg Loss: {epoch_loss:.4f}")
        
        # Save Checkpoint
        save_path = Config.CHECKPOINT_SAVE_DIR / f'{model_name}_epoch_{epoch+1}.pth'
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'num_classes': num_classes,
            'fc_dim': out_dim,
            'backbone': model_name
        }, save_path)
        logger.info(f"Saved checkpoint to {save_path}")

if __name__ == '__main__':
    train_image_model()
